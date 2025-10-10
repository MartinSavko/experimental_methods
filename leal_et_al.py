#!/usr/bin/env python
# conding: utf-8

from omega_scan import omega_scan


class leal_et_al(omega_scan):
    specific_parameter_fields = [
        {
            "name": "nrepeats",
            "type": "int",
            "description": "number of repeats",
        },
        {
            "name": "dosing_factor",
            "type": "int",
            "description": "dosing factor",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        nrepeats=11,
        dosing_factor=10,
        position=None,
        scan_range=5,
        scan_exposure_time=0.5,
        scan_start_angle=None,
        angle_per_frame=0.1,
        nimages_per_file=None,
    ):
        self.default_experiment_name = f"Experimental determination of diffraction intensity decay parameters according to Leal et al. 2010"

        self.nrepeats = nrepeats
        self.dosing_factor = dosing_factor

        if nimages_per_file is None:
            nimages_per_file = scan_range / angle_per_frame

        omega_scan.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
            position=position,
            scan_range=scan_range,
            scan_exposure_time=scan_exposure_time,
            scan_start_angle=scan_start_angle,
            angle_per_frame=angle_per_frame,
            nimages_per_file=nimages_per_file,
        )

        self.md_task_info = []

    def get_nimages(self, epsilon=1e-3):
        _nimages = int(self.scan_range / self.angle_per_frame)
        if abs(nimages * self.angle_per_frame - self.scan_range) > epsilon:
            _nimages += 1
        nimages = (2 * self.nrepeats - 1) * _nimages
        return nimages

    def _measure(self, wait=True):
        task_id = self.goniometer.omega_scan(
            self.scan_start_angle, self.scan_range, self.scan_exposure_time, wait=wait
        )
        self.md_task_info.append(self.goniometer.get_task_info(task_id))

    def _dose(self, wait=True):
        task_id = self.goniometer.omega_scan(
            self.scan_start_angle,
            self.scan_range,
            self.scan_exposure_time * self.dosing_factor,
            wait=wait,
        )
        self.md_task_info.append(self.goniometer.get_task_info(task_id))

    def run(self):
        self._measure()
        for k in range(self.nrepeats - 1):
            self._dose()
            self._measure()
