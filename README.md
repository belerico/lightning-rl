# Advantage Actor-Critic (A2C) Lightning Demo

1. Get [poetry](https://python-poetry.org/docs/#installation)
2. Run `poetry install`
3. Run `lightning run app main.py`

# TODOs

* Add architecture schema drawing
* Add hydra configs for Agent, Model and gym Player (maybe i can avoid the json serializable error...)
* Implement the Trainer class. Given the observed data, read from the Path file:
  1. Compute the A2C loss
  2. Compute gradients
  3. Update model
  4. Share the updated model to the Player, which can collect data with the new policy
* Multiprocessing: 
  1. N Gym Players
  2. N Trainers. Every trainer has its own model, which is updated given the experience gathered by its corresponding Player 
* Move Flow from main.py to its own class
* The Agent class should extend from LightningModule
