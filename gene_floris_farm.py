import floris.tools as wfct
import numpy as np
import matplotlib.pyplot as plt
import csv

fi = wfct.floris_interface.FlorisInterface("example_input.json")

def generation(wind_speed):
    fi.reinitialize_flow_field(wind_speed=wind_speed)
    flow_field = []
    for i in range(100):
        flow_field.append(np.zeros((n_case, 1500)))

    fi.calculate_wake(yaw_angles=yaw_angle[0])
    hor_plane = fi.get_hor_plane(x_resolution=570, y_resolution=310)
    fig, ax = plt.subplots()
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax, cmap='jet')
    plt.show()

    u_mesh = hor_plane.df.u.values.reshape(
        hor_plane.resolution[1], hor_plane.resolution[0]
    )
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)
    for x in range(10):
        for y in range(10):
            flow_field[x * 10 + y] = Zm[5 + 30 * x:5 + 30 * (x + 1), 10 + 50 * y:10 + 50 * (y + 1)].reshape(1500)

    file_name = './generation/U' + str(wind_speed) + '_windfarm.csv'
    with open(file_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(i) for i in range(1500)])
        writer.writerows(i for i in flow_field)


if __name__ == '__main__':
    path = './'
    n_case = 1
    yaw_angle = np.random.uniform(low=-20.0, high=20.0, size=(n_case, 100))

    Horizontal = []
    Vertical = []
    Vertical_unit = [126.4 * 3 * i for i in range(0, 10)]

    for i in range(10):
        Horizontal.extend(10 * [126.4 * 5 * i])
        Vertical.extend(Vertical_unit)

    fi.reinitialize_flow_field(
        layout_array=[Horizontal, Vertical]
    )

    generation(wind_speed=9)
