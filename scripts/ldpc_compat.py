"""Explicit adapter; record the backend version with every experiment."""
import ldpc

def BpOsdDecoder(pcm, *, error_channel, max_iter=30,
                 bp_method='minimum_sum', ms_scaling_factor=1.0,
                 osd_method='OSD_0', osd_order=0, omp_thread_count=1):
    if hasattr(ldpc, 'BpOsdDecoder'):
        return ldpc.BpOsdDecoder(pcm, error_channel=error_channel,
            max_iter=max_iter, bp_method=bp_method,
            ms_scaling_factor=ms_scaling_factor, osd_method=osd_method,
            osd_order=osd_order, omp_thread_count=omp_thread_count)
    if omp_thread_count != 1:
        raise ValueError('Legacy LDPC uses one thread per decoder')
    return ldpc.bposd_decoder(pcm, channel_probs=error_channel,
        max_iter=max_iter,
        bp_method={'minimum_sum':'ms','product_sum':'ps'}.get(bp_method,bp_method),
        ms_scaling_factor=ms_scaling_factor,
        osd_method={'OSD_0':'osd_0','OSD_E':'osd_e','OSD_CS':'osd_cs'}.get(osd_method,osd_method),
        osd_order=osd_order)
