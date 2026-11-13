from .config import normalized
from p2_ltx_all_lora import data as native
from p2_ltx_all_lora.data import to_device
def Dataset(c):return native.Dataset(normalized(c))
def data_identity(c):return native.data_identity(normalized(c))
def data_config(c):return native.data_config(normalized(c))
