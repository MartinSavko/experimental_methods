#!/usr/local/conda/envs/murko_3.11/bin/python

from beam_center import beam_center

from speech import speech, defer

from useful_routines import DEFAULT_BROKER_PORT

class speaking_beam_center(beam_center, speech):
    
    def __init__(
        self,
        name="beam_center",
        service="beam_center",
        port=DEFAULT_BROKER_PORT,
        verbose=None,
        server=None,
    ):
      
        beam_center.__init__(self)
            
        speech.__init__(
            self,
            port=port,
            service=service,
            verbose=verbose,
            server=server,
        )
        
    @defer
    def get_beam_center(
        self,
        wavelength=None,
        ts=None,
        tx=None,
        tz=None,
        ts_offset=0,
        tx_offset=20.5,
        tz_offset=40.5,
        manual_horizontal_offset=0,
        manual_vertical_offset=0,
        wavelength_delta=0.025,
    ):
        return super().get_beam_center(
            wavelength=wavelength,
            ts=ts,
            tx=tx,
            tz=tz,
            ts_offset=ts_offset,
            tx_offset=tx_offset,
            tz_offset=tz_offset,
            manual_horizontal_offset=manual_horizontal_offset,
            manual_vertical_offset=manual_vertical_offset,
            wavelength_delta=wavelength_delta,
        )

if __name__ == "__main__":
    import gevent
    sbc = speaking_beam_center()
    sbc.verbose = True
    #sbc.serve()
    while sbc.server:
        gevent.sleep(0.1)
    
