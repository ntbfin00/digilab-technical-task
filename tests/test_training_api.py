"""HTTP matrix and lifespan tests; W&B is always mocked."""

from dataclasses import replace
from io import BytesIO
import json
import os
from threading import get_ident
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from app.api.routes.training import training_service
from app.main import app
from app.services.data_service import DataService, DatasetTooLargeError
from app.services.training_service import TrainingRun, TrainingRunNotFoundError, TrainingService
from test_data_service import malformed_npz_archives


class TrainingAPITests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.is_shutdown = False
        self.run = TrainingRun(uuid4(), uuid4(), total_steps=100)
        self.service.submit.return_value = self.run
        self.service.submit_prepared.return_value = self.run
        self.service.get_progress.return_value = self.run
        app.dependency_overrides[training_service] = lambda: self.service
        self.client = TestClient(app)
        self.body = {"input_features": [[1, 2], [3, 4]], "target": [[5], [6]],
                     "input_feature_labels": ["a", "b"]}
        self.wandb = patch("app.services.training_service.wandb.Api").start()
        self.init = patch("app.services.training_service.wandb.init").start()

    def tearDown(self):
        app.dependency_overrides.clear()
        patch.stopall()

    def archive(self):
        output = BytesIO()
        np.savez_compressed(output, input_features=self.body["input_features"],
                            target=self.body["target"],
                            input_feature_labels=np.array(self.body["input_feature_labels"]))
        return output.getvalue()

    def test_json_and_upload_same_schema_location_and_config(self):
        first = self.client.post("/training-runs", json=self.body)
        second = self.client.post("/training-runs/upload", files={"file": ("data.npz", self.archive())},
                                  data={"config": '{"epochs": 4}'})
        for response in (first, second):
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.headers["location"], f"http://testserver/training-runs/{self.run.model_id}")
            self.assertEqual(response.json()["model_id"], str(self.run.model_id))
        self.assertEqual(first.json(), second.json())
        self.assertEqual(self.service.submit.call_args.args[3].epochs, 100)
        self.assertEqual(self.service.submit_prepared.call_args.args[1].epochs, 4)
        np.testing.assert_array_equal(self.service.submit.call_args.args[0],
                                      self.service.submit_prepared.call_args.args[0].training_data.input_features)
        decoded_target = self.service.submit_prepared.call_args.args[0].training_data.target.values
        np.testing.assert_array_equal(decoded_target, self.body["target"])
        np.testing.assert_array_equal(decoded_target, self.service.submit.call_args.args[1])

    def test_locations_include_mount_and_root_path_and_can_be_followed(self):
        parent = FastAPI()
        parent.mount("/models", app)
        with TestClient(parent, root_path="/gateway") as client:
            for path, kwargs in (("/training-runs", {"json": self.body}),
                                 ("/training-runs/upload", {"files": {"file": self.archive()}})):
                with self.subTest(path=path):
                    response = client.post(f"/gateway/models{path}", **kwargs)
                    self.assertEqual(response.status_code, 202, response.text)
                    location = response.headers["location"]
                    self.assertEqual(location, f"http://testserver/gateway/models/training-runs/{self.run.model_id}")
                    progress = client.get(location)
                    self.assertEqual(progress.status_code, 200, progress.text)
                    self.assertEqual(progress.json()["model_id"], str(self.run.model_id))

    def test_integer_overflow_in_arrays_and_config_returns_422(self):
        for value in (10**400, -(10**400)):
            for field in ("input_features", "target", "learning_rate", "validation_fraction"):
                with self.subTest(field=field, value=value):
                    change = ({field: [[value], [1]]} if field in {"input_features", "target"}
                              else {"config": {field: value}})
                    response = self.client.post("/training-runs", json=self.body | change)
                    self.assertEqual(response.status_code, 422, response.text)
            for field in ("learning_rate", "validation_fraction"):
                response = self.client.post("/training-runs/upload", files={"file": self.archive()},
                                            data={"config": json.dumps({field: value})})
                self.assertEqual(response.status_code, 422, response.text)
        self.service.submit.assert_not_called()
        self.service.submit_prepared.assert_not_called()

    def test_malformed_unicode_and_lzma_uploads_return_422(self):
        for kind, payload in malformed_npz_archives().items():
            with self.subTest(kind=kind):
                response = self.client.post("/training-runs/upload", files={"file": ("bad.npz", payload)})
                self.assertEqual(response.status_code, 422, response.text)
        self.service.submit_prepared.assert_not_called()

    def test_strict_json_and_config_errors(self):
        changes = [{"input_features": [[True], [1]]}, {"input_features": [["1"], [2]]},
                   {"input_feature_labels": [1, 2]}, {"config": {"epochs": True}},
                   {"config": {"epochs": "2"}}, {"config": {"learning_rate": "0.1"}},
                   {"config": {"validation_fraction": 1}}, {"config": {"seed": -1}},
                   {"config": {"extra": 1}}]
        for change in changes:
            with self.subTest(change=change):
                self.assertEqual(self.client.post("/training-runs", json=self.body | change).status_code, 422)
        for value in ("NaN", "Infinity", "-Infinity"):
            body = json.dumps(self.body).replace("[[1, 2]", f"[[{value}, 2]")
            self.assertEqual(self.client.post("/training-runs", content=body,
                                             headers={"content-type": "application/json"}).status_code, 422)
        self.service.submit.assert_not_called()

    def test_invalid_data_archive_and_config_do_not_accept(self):
        for data in (b"garbage", self.archive()):
            config = "{}" if data == b"garbage" else "not json"
            response = self.client.post("/training-runs/upload", files={"file": ("bad.npz", data)},
                                        data={"config": config})
            self.assertEqual(response.status_code, 422)
        self.service.submit_prepared.assert_not_called()
        for error in (ValueError("ragged"), TypeError("invalid data")):
            self.service.submit.side_effect = error
            self.assertEqual(self.client.post("/training-runs", json=self.body).status_code, 422)

    def test_upload_too_large_no_submission(self):
        for limits in ({"max_compressed_bytes": 1}, {"max_expanded_bytes": 1}):
            with patch("app.api.routes.training.DataService", return_value=DataService(**limits)):
                response = self.client.post("/training-runs/upload", files={"file": ("big.npz", self.archive())})
            self.assertEqual(response.status_code, 413)
        self.service.submit_prepared.assert_not_called()

    def test_local_progress_and_failed_details_do_not_query_remote(self):
        for status in ("running", "completed", "failed"):
            self.service.get_progress.return_value = replace(
                self.run, status=status, completed_steps=3, training_loss=1.5,
                validation_loss=2.5, best_validation_loss=2.0, message="failure detail",
                run_url="https://example/run", artifact_reference="team/project/model:v0")
            response = self.client.get(f"/training-runs/{self.run.model_id}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], status)
            self.assertEqual(response.json()["completed_steps"], 3)
            self.assertEqual(response.json()["validation_loss"], 2.5)
            self.assertEqual(response.json()["message"], "failure detail")
            self.assertEqual(response.json()["artifact_reference"], "team/project/model:v0")
        self.service.get_historical_progress.assert_not_called()
        self.wandb.assert_not_called()

    def test_local_miss_fallback_missing_outage_and_bad_metadata(self):
        historical = replace(self.run, status="completed", completed_steps=100,
                             training_loss=1.25, validation_loss=2.5, best_validation_loss=2.0,
                             run_url="https://example/historical", artifact_reference="team/project/model:v1")

        def local_lookup(model_id):
            self.assertEqual(model_id, historical.model_id)
            raise TrainingRunNotFoundError()

        def historical_lookup(model_id):
            self.assertEqual(model_id, historical.model_id)
            return historical

        self.service.get_progress.side_effect = local_lookup
        self.service.get_historical_progress.side_effect = historical_lookup
        path = f"/training-runs/{self.run.model_id}"
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200)
        self.service.get_progress.assert_called_once_with(historical.model_id)
        self.service.get_historical_progress.assert_called_once_with(historical.model_id)
        self.assertEqual(response.json(), {
            "model_id": str(historical.model_id), "dataset_id": str(historical.dataset_id),
            "status": "completed", "completed_steps": 100, "total_steps": 100, "step_unit": "epoch",
            "training_loss": 1.25, "validation_loss": 2.5, "best_validation_loss": 2.0,
            "message": None, "run_url": "https://example/historical",
            "artifact_reference": "team/project/model:v1",
        })
        self.assertEqual([c[0] for c in self.service.mock_calls], ["get_progress", "get_historical_progress"])
        self.service.get_historical_progress.side_effect = TrainingRunNotFoundError()
        self.assertEqual(self.client.get(path).status_code, 404)
        for error in (RuntimeError("secret outage"), PermissionError("secret credentials"),
                      ValueError("bad metadata")):
            self.service.get_historical_progress.side_effect = error
            with self.assertLogs("app.api.routes.training", level="ERROR"):
                response = self.client.get(path)
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("secret", response.text)
        self.service.get_historical_progress.side_effect = None
        self.service.get_historical_progress.return_value = replace(self.run, status="invalid")
        with self.assertLogs("app.api.routes.training", level="ERROR"):
            self.assertEqual(self.client.get(path).status_code, 503)

    def test_bad_uuid_does_not_lookup(self):
        self.assertEqual(self.client.get("/training-runs/not-a-uuid").status_code, 422)
        self.service.get_progress.assert_not_called()

    def test_shutdown_submission_returns_503(self):
        self.service.submit.side_effect = RuntimeError("shut down")
        self.service.submit_prepared.side_effect = RuntimeError("shut down")
        with self.assertLogs("app.api.routes.training", level="ERROR"):
            self.assertEqual(self.client.post("/training-runs", json=self.body).status_code, 503)
            self.assertEqual(self.client.post("/training-runs/upload", files={"file": self.archive()}).status_code, 503)

    def test_closed_service_dependency_rejects_all_training_routes(self):
        app.dependency_overrides.clear()
        service = TrainingService(project="test", entity="team")
        service.shutdown()
        with patch.object(app.state, "training_service", service, create=True):
            self.assertEqual(self.client.get(f"/training-runs/{uuid4()}").status_code, 503)
            self.assertEqual(self.client.post("/training-runs", json=self.body).status_code, 503)
            self.assertEqual(self.client.post("/training-runs/upload", files={"file": self.archive()}).status_code, 503)
            self.assertEqual(self.client.get("/").status_code, 200)

    def test_lifespan_missing_config_health_and_shared_service_cleanup(self):
        app.dependency_overrides.clear()
        with patch.dict(os.environ, {"WANDB_PROJECT": "", "WANDB_ENTITY": ""}):
            with TestClient(app) as client:
                self.assertEqual(client.get("/").json(), {"status": "running"})
                self.assertEqual(client.post("/training-runs", json=self.body).status_code, 503)
                self.assertEqual(client.post("/training-runs/upload", files={"file": self.archive()}).status_code, 503)
                self.assertEqual(client.get(f"/training-runs/{uuid4()}").status_code, 503)
        threads = {}
        with patch.dict(os.environ, {"WANDB_PROJECT": "project", "WANDB_ENTITY": "team"}), \
                patch("app.main.TrainingService", return_value=self.service) as factory:
            self.service.shutdown.side_effect = lambda: threads.update(shutdown=get_ident())
            with TestClient(app) as client:
                client.portal.call(lambda: threads.update(event_loop=get_ident()))
                self.service.submit.side_effect = lambda *args: (threads.update(submit=get_ident()) or self.run)
                self.assertEqual(client.post("/training-runs", json=self.body).status_code, 202)
                self.assertIs(app.state.training_service, self.service)
            factory.assert_called_once_with(project="project", entity="team")
        self.assertIsNone(app.state.training_service)
        self.service.shutdown.assert_called_once()
        self.assertNotEqual(threads["shutdown"], threads["event_loop"])
        self.assertNotEqual(threads["submit"], threads["event_loop"])
        self.init.assert_not_called()

    def test_openapi_contract(self):
        schema = app.openapi()
        paths = schema["paths"]
        post = paths["/training-runs"]["post"]
        self.assertIn("application/json", post["requestBody"]["content"])
        upload = paths["/training-runs/upload"]["post"]
        multipart = upload["requestBody"]["content"]["multipart/form-data"]["schema"]
        fields = schema["components"]["schemas"][multipart["$ref"].split("/")[-1]]["properties"]
        self.assertEqual(set(fields), {"file", "config"})
        self.assertEqual(post["responses"]["202"]["content"], upload["responses"]["202"]["content"])
        get = paths["/training-runs/{model_id}"]["get"]
        self.assertEqual(get["parameters"][0]["schema"]["format"], "uuid")
        self.assertIn("200", get["responses"])
        for operation, codes in ((post, (503,)), (upload, (413, 503)), (get, (404, 503))):
            for code in codes:
                error = operation["responses"][str(code)]["content"]["application/json"]["schema"]
                model = schema["components"]["schemas"][error["$ref"].split("/")[-1]]
                self.assertEqual(model["properties"]["detail"]["type"], "string")
                self.assertIn("detail", model["required"])
        for operation in (post, upload):
            location = operation["responses"]["202"]["headers"]["Location"]
            self.assertEqual(location["schema"], {"type": "string", "format": "uri"})
            self.assertIn("polling", location["description"])

    def test_real_service_shared_submission_validation_and_preserved_ids(self):
        service = TrainingService(project="test", entity="team")
        app.dependency_overrides[training_service] = lambda: service
        # Keep accepted jobs queued; do not execute W&B or optimization here.
        with patch.object(service._worker, "submit"), \
                patch.object(service, "submit_prepared", wraps=service.submit_prepared) as submit:
            try:
                first = self.client.post("/training-runs", json=self.body)
                second = self.client.post("/training-runs/upload", files={"file": self.archive()})
                self.assertEqual(first.status_code, 202)
                self.assertEqual(second.status_code, 202)
                self.assertEqual(submit.call_count, 2)
                for response, call in zip((first, second), submit.call_args_list):
                    self.assertEqual(response.json()["dataset_id"], str(call.args[0].dataset_id))
                    self.assertEqual(self.client.get(response.headers["location"]).status_code, 200)
                self.assertNotEqual(first.json()["model_id"], second.json()["model_id"])
                for changes in ({"input_features": [[1], [2, 3]]}, {"target": [[1]]},
                                {"input_feature_labels": ["only one"]},
                                {"input_features": [[1, 2]], "target": [[1]]}):
                    self.assertEqual(self.client.post("/training-runs", json=self.body | changes).status_code, 422)
                self.assertEqual(len(service._runs), 2)
                self.init.assert_not_called()
            finally:
                service.shutdown()
