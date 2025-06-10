from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
        flow_location="score_prefect.py",
        name="ride_duration_prediction",
        parameters={
            "run_id":"54c0663c677e417f8900b51ed7985878"
            },
        flow_storage="1c5c036d-913a-40c4-910c-1339c086fbba", 
        schedule=CronSchedule(cron="0 3 2 * *"),
        flow_runner=SubprocessFlowRunner(),
        tags=["ml"]
        )
