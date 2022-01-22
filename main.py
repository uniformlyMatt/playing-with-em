import numpy as np
import plotly.graph_objects as go
import utils

if __name__ == '__main__':
    # define the domain
    x = np.linspace(0, 10, 101)

    # generate synthetic samples
    y1 = utils.gaussian(x, mu=2, sigma_squared=.95)
    y2 = utils.gaussian(x, mu=7, sigma_squared=.95)

    result = utils.fit(y1, y2, pi_initial=.5, tolerance=.0001)
    print(result[1:])

    # fig = go.Figure()
    #
    # fig.add_trace(
    #     go.Scatter(
    #         x=x,
    #         y=y1,
    #         mode='lines'
    #     )
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=x,
    #         y=y2,
    #         mode='lines'
    #     )
    # )
    # fig.show()
