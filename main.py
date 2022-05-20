import lightning as L

from demos.a2c_demo.gym.player import GymWorker
from demos.a2c_demo.trainer.trainer import TrainWorker


class TrainDeploy(L.LightningFlow):
    def __init__(self, worker: GymWorker, trainer: TrainWorker):
        super().__init__()
        self.gym_worker = worker
        self.train_worker = trainer

    def run(self):
        self.gym_worker.run()
        self.train_worker.run()


if __name__ == "__main__":
    worker = GymWorker("LunarLander-v2")
    trainer = TrainWorker(agent_data_path=worker.output_data_path, agent_sizes_path=worker.output_sizes_path)
    app = L.LightningApp(TrainDeploy(worker, trainer))
