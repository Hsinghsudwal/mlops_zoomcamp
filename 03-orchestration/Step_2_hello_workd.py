from prefect import flow, task

@task
def hello_prefect():
    print("Hello, World!")

@flow
def hello_world_flow():
    hello_prefect()

if __name__ == "__main__":
    hello_world_flow()
