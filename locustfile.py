from locust import HttpUser, task, between

class LLMUserSimulation(HttpUser):
    # Simulate a user waiting between 0.1 tp 0.5 seconds between clicks
    wait_time = between(0.1, 0.5)

    @task
    def hit_llm_endpoint(self):
        headers = {"Content-Type": "application/json"}
        payload = {"prompt": "Testing high load scalability"}

        # Fire post request for fast api
        self.client.post("/v1/generate", json=payload, headers=headers)