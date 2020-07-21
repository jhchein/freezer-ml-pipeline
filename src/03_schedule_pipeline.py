from azureml.core import Experiment, Workspace
from azureml.pipeline.core import PublishedPipeline, Schedule, ScheduleRecurrence

ws = Workspace.from_config()
exp = Experiment(ws, "MaxFreezerTemperatureExceededPipeline", _create_in_cloud=True)
pipeline_id = PublishedPipeline.list(ws)[0]

schedule = Schedule.create(
    ws,
    name="four_updates_per_day",
    description="runs the pipeline every 6 hours",
    pipeline_id=pipeline_id,
    recurrence=ScheduleRecurrence(
        frequency="Hour",
        interval=6,
        start_time=None,  # run instantly
        time_zone=None,  # default UTC
    ),
    experiment_name=exp.name,
)

# Schedule.list(ws)
# schedule = Schedule.list(ws)[0]
# schedule.get_last_pipeline_run()
