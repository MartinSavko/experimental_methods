from goniometer import goniometer

from detector import detector

from camera import camera

from instrument import instrument

from cats import cats

from beam_center import beam_center

from fast_shutter import fast_shutter

from safety_shutter import safety_shutter

from frontend_shutter import frontend_shutter

from monitor import xray_camera, Si_PIN_diode

from transmission import transmission

from flux import flux

from mirror_scan import adaptive_mirror

from fluorescence_detector import fluorescence_detector

from energy import energy as photon_energy

from focusing import focusing

from machine_status import machine_status

from resolution import resolution

from energy import energy

from beam_position_controller import get_bpc

g = goniometer()
d = detector()
cam = camera()
c = cats()
i = instrument()
bc = beam_center()
saf = safety_shutter()
front = frontend_shutter()
fs = fast_shutter()
pin = Si_PIN_diode()
xc = xray_camera()
t = transmission()
f = flux()
vfm = adaptive_mirror('vfm')
hfm = adaptive_mirror('hfm')
fd = fluorescence_detector()
pe = photon_energy()
focus = focusing()
mach = machine_status()
res = resolution()
en = energy()

vbpc = get_bpc(monitor='cam', actuator='vertical_trans', period=0.25, ponm=False)
hbpc = get_bpc(monitor='cam', actuator='horizontal_trans', period=0.25, ponm=False)
