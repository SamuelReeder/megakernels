"""Spike only (disposable). Host side of the DSv4.1 peer memory, copied without vLLM:

- ``UncachedIpcHeap``: aiter v0.1.26 ``aiter/ops/flydsl/quick_allreduce_int4_ipc.py``
  (the methods ``PeerBuffer`` uses), with aiter's ``load_hip_runtime``.
- ``PeerBuffer``: vLLM ``models/deepseek_v41/amd/mono/common/peer_memory.py``.
- ``tp_agree``: vLLM ``models/common/specialized_decode/guards.py``.
The container has no aiter and no vLLM; the protocol is unchanged.
"""

import ctypes

import torch
import torch.distributed as dist


def load_hip_runtime():
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise OSError("no HIP runtime")


class UncachedIpcHeap:
    _HIP_IPC_HANDLE_BYTES = 64
    _LAZY_PEER = 0x1
    _UNCACHED = 0x3
    _hip = None

    class _Handle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_byte * 64)]

    @classmethod
    def _load(cls):
        if cls._hip is None:
            h = load_hip_runtime()
            H = cls._Handle
            h.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(H), ctypes.c_void_p]
            h.hipIpcOpenMemHandle.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                H,
                ctypes.c_uint,
            ]
            h.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
            h.hipExtMallocWithFlags.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
                ctypes.c_uint,
            ]
            h.hipFree.argtypes = [ctypes.c_void_p]
            h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
            h.hipGetErrorString.restype = ctypes.c_char_p
            cls._hip = h
        return cls._hip

    @classmethod
    def _check(cls, err, what):
        if err:
            raise RuntimeError(f"{what}: {cls._load().hipGetErrorString(err)}")

    @classmethod
    def get_mem_handle_bytes(cls, ptr):
        h = cls._Handle()
        cls._check(cls._load().hipIpcGetMemHandle(ctypes.byref(h), ptr), "IpcGet")
        return bytes(ctypes.string_at(ctypes.byref(h), 64))

    @classmethod
    def open_mem_handle(cls, b):
        h = cls._Handle()
        ctypes.memmove(ctypes.byref(h), b, 64)
        out = ctypes.c_void_p()
        cls._check(
            cls._load().hipIpcOpenMemHandle(ctypes.byref(out), h, cls._LAZY_PEER),
            "IpcOpen",
        )
        return int(out.value)

    @classmethod
    def close_mem_handle(cls, ptr):
        cls._check(cls._load().hipIpcCloseMemHandle(ctypes.c_void_p(ptr)), "IpcClose")

    @classmethod
    def alloc_uncached(cls, size):
        size = (size + 4095) // 4096 * 4096
        buf = ctypes.c_void_p()
        hip = cls._load()
        cls._check(hip.hipExtMallocWithFlags(ctypes.byref(buf), size, cls._UNCACHED), "malloc")
        cls._check(hip.hipMemset(buf, 0, size), "memset")
        return int(buf.value)

    @classmethod
    def free_device_mem(cls, ptr):
        cls._check(cls._load().hipFree(ctypes.c_void_p(ptr)), "free")

    @staticmethod
    def gather_object_list_via_broadcast(group, shard):
        ws, rank = dist.get_world_size(group), dist.get_rank(group)
        data = [[None] for _ in range(ws)]
        data[rank][0] = shard
        for i, r in enumerate(dist.get_process_group_ranks(group)):
            dist.broadcast_object_list(data[i], src=r, group=group, device="cpu")
        return [d[0] for d in data]


def tp_agree(ok, group):
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


class _DeviceBytes:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


class PeerBuffer:
    def __init__(self, nbytes, group, rank, npes, device):
        self.local = UncachedIpcHeap.alloc_uncached(nbytes)
        self.bytes = torch.as_tensor(_DeviceBytes(self.local, nbytes), device=device)
        self.rank = rank
        self._opened = []
        addresses = [self.local]
        if npes > 1:
            handles = UncachedIpcHeap.gather_object_list_via_broadcast(
                group, UncachedIpcHeap.get_mem_handle_bytes(self.local)
            )
            addresses, failure = [], None
            try:
                for peer, handle in enumerate(handles):
                    if peer == rank:
                        addresses.append(self.local)
                    else:
                        base = UncachedIpcHeap.open_mem_handle(handle)
                        self._opened.append(base)
                        addresses.append(base)
            except RuntimeError as err:
                failure = err
            if not tp_agree(failure is None, group):
                self.close()
                raise RuntimeError("peer memory handshake failed") from failure
        self.addresses = torch.tensor(addresses, dtype=torch.int64, device=device)

    def close(self):
        opened, self._opened = self._opened, []
        for base in opened:
            UncachedIpcHeap.close_mem_handle(base)
        if self.local:
            self.bytes = None
            UncachedIpcHeap.free_device_mem(self.local)
            self.local = 0
