"""PFN conditions include a read-only copy of the native action center."""
import numpy as np


PARENT_READOUT_ID = 'uniwam-robotwin.all-vlm-adapter-mean+prepared-input-state.native-action-epsilon.v1'
HISTORY_READOUT_ID = 'uniwam-robotwin.all-vlm-adapter-mean+prepared-input-state+native-action-source.native-action-epsilon.v2'
JOINT_PARENT_READOUT_ID = 'uniwam-robotwin.all-vlm-adapter-mean+prepared-input-state.native-history-epsilon.v3'
JOINT_HISTORY_READOUT_ID = 'uniwam-robotwin.all-vlm-adapter-mean+prepared-input-state+native-action-source.native-history-epsilon.v4'


def append_action_history(features, native_action_source):
    """Append the observed cast H; never construct, normalize or replace it."""
    features = np.asarray(features)
    history = np.asarray(native_action_source)
    if (features.ndim != 2 or history.ndim != 3 or len(features) != len(history)
            or features.dtype != np.float32 or history.dtype != np.float32
            or not np.isfinite(features).all() or not np.isfinite(history).all()):
        raise ValueError('Use finite original features and the actual native history readout.')
    return np.concatenate((features, history.reshape(len(history), -1)), axis=1)
