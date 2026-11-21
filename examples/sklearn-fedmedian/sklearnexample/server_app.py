"""sklearnexample: A Flower / sklearn app."""

import joblib
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedMedian
from flwr_viz import instrument  # flwr-viz

from sklearnexample.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    set_model_params,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)
    # Construct ArrayRecord representation
    arrays = ArrayRecord(get_model_params(model))

    # FedMedian (coordinate-wise median) instead of the template's FedAvg,
    # sampling a fraction of nodes each round, wrapped by flwr-viz.
    strategy = instrument(
        FedMedian(fraction_train=context.run_config["fraction-train"], fraction_evaluate=0.5,
                  min_available_nodes=context.run_config["min-available-clients"]),
        run_config=context.run_config,
    )

    # Start strategy, run FedMedian for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    if context.run_config["save-model"]:
        # Save final model parameters
        print("\nSaving final model to disk...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        set_model_params(model, ndarrays)
        joblib.dump(model, "logreg_model.pkl")
