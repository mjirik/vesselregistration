import pytest
import vesselregistration.moduleone
import vesselregistration.registration
import numpy as np

def test_hello():
    vesselregistration.moduleone.print_hello("Vlkoslav")


def test_vr():
    volume1 = np.zeros([50, 51, 52])
    volume1[30:40, 30:40, 10:40] = 1
    volume1[30:38, 10:38, 30:38] = 1

    volume2 = np.zeros([500, 510, 520])
    volume2[300:400, 300:400, 100:400] = 1
    volume2[300:380, 100:380, 300:380] = 1

    registrator = vesselregistration.registration.VesselRegistraion()
    tt1 = registrator.create_tube_table(volume1)
    tt2 = registrator.create_tube_table(volume2)

    kp1 = registrator.create_keypoint_table(tt1)
    kp2 = registrator.create_keypoint_table(tt2)

    tfmatrix = registrator.registration(kp1, kp2)
    pass