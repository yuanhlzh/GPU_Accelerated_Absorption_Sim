
#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import os
import platform
import matplotlib.pyplot as plt
import pathlib

class Gaas_OCL_API:
    def __init__(self) -> None:
        self.dev = None
        self.ctx = None
        self.queue = None
        self.mf = cl.mem_flags
        self.prg = None
        self.knl_lineshapeSDV = None
        self.use_opencl = False
        # self.select_opencl_device(oclDevice=oclDevice)

    @classmethod
    def list_compute_devices(cls):
        """
        Returns all devices the user may select in the UI:
        - OpenCL GPUs
        - OpenCL CPUs
        - Native CPU fallback
        """
        devices = []

        opencl_devices = cls.list_opencl_devices()

        # Normalize OpenCL entries
        for dev in opencl_devices:
            if dev.get("platform_index") is None:
                continue

            dev = dict(dev)
            dev["backend"] = "opencl"
            dev["is_opencl"] = True

            # make OpenCL explicit in the label
            if "OpenCL" not in dev["label"]:
                dev["label"] = f'{dev["label"]} | OpenCL'

            devices.append(dev)

        # Add a normal CPU entry even if OpenCL CPU is unavailable
        host_cpu = cls._get_host_cpu_info()

        # Avoid duplicate-looking CPU entries if an OpenCL CPU is already present
        has_opencl_cpu = any(
            d.get("device_type") == "CPU" and d.get("is_opencl") for d in devices
        )

        if has_opencl_cpu:
            host_cpu["label"] = (
                f'CPU | {host_cpu["device_name"]} | Native CPU (no OpenCL acceleration)'
            )

        devices.append(host_cpu)

        return devices

    @staticmethod
    def list_opencl_devices():
        """
        Returns all available OpenCL devices (CPU + GPU) across all platforms.

        Output format:
        [
            {
                "label": "...",
                "platform_index": int,
                "device_index": int,
                "device_type": "GPU" or "CPU",
                "platform_name": str,
                "device_name": str
            }
        ]
        """

        devices = []

        try:
            platforms = cl.get_platforms()
        except Exception as exc:
            return [{
                "label": f"No OpenCL platforms found: {exc}",
                "platform_index": None,
                "device_index": None,
                "device_type": None,
                "platform_name": None,
                "device_name": None,
            }]

        for p_idx, platform in enumerate(platforms):

            try:
                platform_devices = platform.get_devices()
            except Exception:
                continue

            for d_idx, dev in enumerate(platform_devices):

                if dev.type & cl.device_type.GPU:
                    dtype = "GPU"
                elif dev.type & cl.device_type.CPU:
                    dtype = "CPU"
                else:
                    dtype = "OTHER"

                try:
                    mem_gb = dev.global_mem_size / (1024 ** 3)
                    mem_txt = f" | {mem_gb:.1f} GB"
                except Exception:
                    mem_txt = ""

                label = f"{dtype} | {platform.name} | {dev.name}{mem_txt}"

                devices.append({
                    "label": label,
                    "platform_index": p_idx,
                    "device_index": d_idx,
                    "device_type": dtype,
                    "platform_name": platform.name,
                    "device_name": dev.name,
                })

        return devices

    @staticmethod
    def _get_host_cpu_info():
        """
        Returns a CPU entry even if no OpenCL CPU runtime is installed.
        """
        cpu_name = platform.processor().strip()
        if not cpu_name:
            cpu_name = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if not cpu_name:
            cpu_name = "System CPU"

        thread_count = os.cpu_count()
        thread_txt = f" | {thread_count} threads" if thread_count is not None else ""

        return {
            "label": f"CPU | {cpu_name}{thread_txt} | Native CPU (no OpenCL acceleration)",
            "platform_index": None,
            "device_index": None,
            "device_type": "CPU",
            "platform_name": None,
            "device_name": cpu_name,
            "backend": "native_cpu",
            "is_opencl": False,
        }

    def select_compute_device(self, device_info: dict):
        """
        Select either an OpenCL device or the native CPU fallback.
        """
        self.selected_device_info = dict(device_info)

        backend = device_info.get("backend")

        if backend == "opencl":
            self.select_opencl_device(
                platform_index=device_info["platform_index"],
                device_index=device_info["device_index"],
            )
            self.use_opencl = True
        elif backend == "native_cpu":
            self.dev = None
            self.ctx = None
            self.queue = None
            self.prg = None
            self.knl_lineshapeSDV = None
            self.use_opencl = False
            print(f"GAAS: Using native CPU backend: {device_info.get('device_name')}")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def select_opencl_device(
        self,
        platform_index=None,
        device_index=None,
        device_type="GPU",
        oclDevice=None,
    ):
        """
        Selects the OpenCL device, creates the context/queue, and rebuilds the program.
        """

        if oclDevice is not None:
            if isinstance(oclDevice, (list, tuple)):
                self.dev = list(oclDevice)
            else:
                self.dev = [oclDevice]
        else:
            selected_platform_index = 1
            selected_device_type = "GPU"

            if os.getenv("GAAS_OCL_DEVICE") is not None:
                selected_platform_index = int(os.getenv("GAAS_OCL_DEVICE"))

            if os.getenv("GAAS_OCL_DEVICE_TYPE") is not None:
                env_type = os.getenv("GAAS_OCL_DEVICE_TYPE")
                if env_type in ("GPU", "CPU"):
                    selected_device_type = env_type

            if platform_index is not None:
                selected_platform_index = int(platform_index)

            if device_type in ("GPU", "CPU"):
                selected_device_type = device_type

            platforms = cl.get_platforms()
            platform = platforms[selected_platform_index]

            cl_type = cl.device_type.CPU if selected_device_type == "CPU" else cl.device_type.GPU
            devs = platform.get_devices(device_type=cl_type)

            if device_index is None:
                self.dev = [devs[0]]
            else:
                self.dev = [devs[int(device_index)]]

        print("GAAS OpenCL: Using ", self.dev)

        self.ctx = cl.Context(devices=self.dev)
        self.queue = cl.CommandQueue(self.ctx)

        thisFilePath = os.path.realpath(os.path.dirname(__file__))

        try:
            with open(os.path.join(thisFilePath, "Gaas.c"), "r") as file:
                srcStr = file.read()
        except IOError:
            print("Failed to load opencl source files")
            raise

        self.prg = cl.Program(self.ctx, srcStr)
        hdrSearchPath = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        self.prg.build(options="-I" + hdrSearchPath + " -cl-finite-math-only -w")

        # print(self.prg.get_build_info(self.ctx.devices[0], cl.program_build_info.LOG))
        # print(self.prg.get_info(cl.program_info.KERNEL_NAMES))

        self.knl_lineshapeSDV = cl.Kernel(self.prg, "lineshapeSDV")
        self.knl_lineshapeVoigt = cl.Kernel(self.prg, "lineshapeVoigt_raw_cmp")

    def runVoigtTest(self, wvn : np.array):
        wvn = wvn.astype(np.float64)
        wvn_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=wvn)
        out_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, wvn.nbytes)
        knl = self.prg.voigtTest
        knl(self.queue,wvn.shape,None,out_g,wvn_g)
        out_np = np.empty_like(wvn)
        cl.enqueue_copy(self.queue, out_np, out_g)
        return out_np

    def getVoigtDBStructDatatype(self):
        """
            Returns the voigt feature database data structure which is compatible with the OpenCL code
        """
        return [('transWavenum','<f8'),('nAir','<f8'),('gammaAir','<f8'),('gammaSelf','<f8'),('refStrength','<f8'),('ePrimePrime','<f8'),('deltaAir','<f8')]
    

    def getVoigtRawStructDatatype(self):
        """
            Returns the voigt raw feature data structure which is compatible with the OpenCL code
            
            integratedArea;
            transWavenum;
            gammaD; //doppler (gaussian) broadening (HWHM)
            gamma0; //lorenzian broadening (HWHM)
        """
        return [('integratedArea','<f8'),('transWavenum','<f8'),('gammaD','<f8'),('gamma0','<f8'), ('YRosen','<f8')]
    
    def getVoigtRawCmpStructDatatype(self):
        """
            Returns the voigt raw feature data structure which is compatible with the OpenCL code
            
            integratedArea;
            transWavenum;
            gammaD; //doppler (gaussian) broadening (HWHM)
            gamma0; //lorenzian broadening (HWHM)
        """
        return [('integratedArea','<f8'),('transWavenum','<f8'),('gammaD','<f8'),('gamma0','<f8'),('delta0','<f8'), ('YRosen','<f8')]

    def getHTPDBStructDatatype(self):
        """
            Returns the HTP feature database numpy data structure which is compatible with the OpenCL code
        """
        return [('transWavenum','<f8'),
                ('Gam0','<f8'),
                ('Gam2','<f8'),
                ('Delta0','<f8'),
                ('Delta2','<f8'),
                ('anuVC','<f8'),
                ('eta','<f8'),
                ('lineIntensity','<f8')]
    
    def getSDVDBStructDatatype(self):
        """
            Returns the SDV feature database numpy data structure which is compatible with the OpenCL code
        """
        return [('transWavenum','<f8'),
                ('Gam0','<f8'),
                ('Gam2','<f8'),
                ('Delta0','<f8'),
                ('Delta2','<f8'),
                ('YRosen','<f8'),
                ('lineIntensity','<f8')]


    def voigtSim(self, 
                featureDatabase : np.array,
                temp : float,
                pressure : float,
                concentration : float,
                tipsRef : float,
                tipsTemp : float,
                wvnStart : float,
                wvnStep : float,
                wvnEnd : float,
                molarMass : float,
                isoAbundance : float,
                wvnGrid = None
                ):
        wvn_np  = np.arange(wvnStart,wvnEnd,wvnStep).astype(np.float64)
        wvn_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, wvn_np.nbytes, hostbuf = wvn_np) #wvns
        abs_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, wvn_np.nbytes) #absorbance
        cl.enqueue_fill_buffer(self.queue,abs_g,np.float64(0),0,wvn_np.nbytes)
        db_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, featureDatabase.nbytes, hostbuf = featureDatabase) #feature DB

        knl = self.prg.lineshapeVoigt
        
        linesperbatch = 6000
        offset = 0
        numFeats = featureDatabase.size
        while(offset<=np.size(featureDatabase)):
            nleft = numFeats-offset
            kernelSize = min(nleft,linesperbatch)
            if(kernelSize == 0):
                break
            knl(self.queue,(kernelSize,),None,
                wvn_g,
                db_g,
                abs_g,
                np.float64(temp),
                np.float64(pressure),
                np.float64(concentration),
                np.float64(tipsRef),
                np.float64(tipsTemp),
                np.float64(wvnStart),
                np.float64(wvnStep),
                np.int32(wvn_np.size),
                np.float64(molarMass),
                np.float64(isoAbundance),
                np.int32(offset))
            
            offset+=linesperbatch

        self.queue.finish()

        abs_np = np.empty_like(wvn_np)
        cl.enqueue_copy(self.queue, abs_np, abs_g)
        return (wvn_np,abs_np)
    
    def voigtSim_raw(self, 
                featureDatabase : np.array,
                wvnStart : float,
                wvnStep : float,
                wvnEnd : float,
                ):
        
        magic_numbers = [2000,6000,8000,16000,32000,64000] #for some reason, running in batches of this many lines at a time is orders of magnitude faster 
        magic_num = 2000
        # for mn in magic_numbers:
            # if(np.size(featureDatabase)>magic_numbers)

        wvn_np  = np.arange(wvnStart,wvnEnd,wvnStep).astype(np.float64)
        wvn_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, wvn_np.nbytes, hostbuf = wvn_np) #wvns
        abs_g = cl.Buffer(self.ctx, self.mf.READ_WRITE, wvn_np.nbytes) #absorbance
        cl.enqueue_fill_buffer(self.queue,abs_g,np.float64(0),0,wvn_np.nbytes)
        db_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, featureDatabase.nbytes, hostbuf = featureDatabase) #feature DB

        knl = self.prg.lineshapeVoigt_raw
        offset = 0
        numFeats = featureDatabase.size
        while(offset<=np.size(featureDatabase)):
            nleft = numFeats-offset

            #find magic number
            magic_num = magic_numbers[0]
            if(nleft > magic_numbers[-1]):
                magic_num = magic_numbers[-1]
            else:
                for i in range(len(magic_numbers)-1):
                    if(nleft >= magic_numbers[i] and nleft <= magic_numbers[i+1]):
                        magic_num = magic_numbers[i]

            kernelSize = min(nleft,magic_num)
            if(kernelSize == 0):
                break
            knl(self.queue,(kernelSize,1),None,
                wvn_g,
                db_g,
                abs_g,
                np.float64(wvnStart),
                np.float64(wvnStep),
                np.int32(wvn_np.size),
                np.int32(offset))
            
            self.queue.finish()
            offset += magic_num
            # print(offset)
        abs_np = np.empty_like(wvn_np)
        cl.enqueue_copy(self.queue, abs_np, abs_g)
        return (wvn_np,abs_np)
    
    def HTPSim( self, 
                featureDatabase : np.array,
                temp : float,
                wvnStart : float,
                wvnStep : float,
                wvnEnd : float,
                molarMass : float
                ):
        
        magic_numbers = [2000,6000,8000,16000,32000,64000] #for some reason, running in batches of this many lines at a time is orders of magnitude faster 
        magic_num = 2000

        wvn_np  = np.arange(wvnStart,wvnEnd,wvnStep).astype(np.float64)

        wvn_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, wvn_np.nbytes, hostbuf = wvn_np) #wvns
        abs_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, wvn_np.nbytes) #absorbance

        cl.enqueue_fill_buffer(self.queue,abs_g,np.float64(0),0,wvn_np.nbytes)
        db_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, featureDatabase.nbytes, hostbuf = featureDatabase) #feature DB

        knl = self.prg.lineshapeHTP

        offset = 0
        numFeats = featureDatabase.size
        while(offset<=np.size(featureDatabase)):
            nleft = numFeats-offset
            #find magic number
            magic_num = magic_numbers[0]
            if(nleft > magic_numbers[-1]):
                magic_num = magic_numbers[-1]
            else:
                for i in range(len(magic_numbers)-1):
                    if(nleft >= magic_numbers[i] and nleft <= magic_numbers[i+1]):
                        magic_num = magic_numbers[i]

            kernelSize = min(nleft,magic_num)
            if(kernelSize == 0):
                break
            knl(self.queue,(kernelSize,1),None,
                wvn_g,
                db_g,
                abs_g,
                np.float64(temp),
                np.float64(wvnStart),
                np.float64(wvnStep),
                np.int32(wvn_np.size),
                np.float64(molarMass),
                np.int32(offset))
            offset+=magic_num
        abs_np = np.empty_like(wvn_np)
        cl.enqueue_copy(self.queue, abs_np, abs_g)
        return (wvn_np,abs_np)

    def SDVSim( self, 
            featureDatabase : np.array,
            temp : float,
            wvn_np: float,
            wvnStart : float,
            wvnStep : float,
            molarMass : float
            ):
    
        magic_numbers = [2000,6000,8000,16000,32000,64000] #for some reason, running in batches of this many lines at a time is orders of magnitude faster 
        magic_num = 2000

        wvn_np = wvn_np.astype(np.float64)
        # wvn_np  = np.arange(wvnStart,wvnEnd,wvnStep).astype(np.float64)

        wvn_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, wvn_np.nbytes, hostbuf = wvn_np) #wvns
        
         # Allocate complex output buffer
        out_np = np.zeros(wvn_np.size, dtype=np.complex128)
        
        out_g = cl.Buffer(self.ctx,
                      self.mf.WRITE_ONLY,
                      out_np.nbytes)

        # abs_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, wvn_np.nbytes) #absorbance

        zero16 = np.zeros(1, dtype=np.complex128)  # 16 bytes = 128 bits
        cl.enqueue_fill_buffer(self.queue, out_g, zero16.tobytes(), 0, out_np.nbytes)

        db_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, featureDatabase.nbytes, hostbuf = featureDatabase) #feature DB

        knl = self.knl_lineshapeSDV

        offset = 0
        numFeats = featureDatabase.size
        while(offset<=np.size(featureDatabase)):
            nleft = numFeats-offset
            #find magic number
            magic_num = magic_numbers[0]
            if(nleft > magic_numbers[-1]):
                magic_num = magic_numbers[-1]
            else:
                for i in range(len(magic_numbers)-1):
                    if(nleft >= magic_numbers[i] and nleft <= magic_numbers[i+1]):
                        magic_num = magic_numbers[i]

            kernelSize = min(nleft,magic_num)
            if(kernelSize == 0):
                break
            knl(self.queue,(kernelSize,1),None,
                wvn_g,
                db_g,
                out_g,
                np.float64(temp),
                np.float64(wvnStart),
                np.float64(wvnStep),
                np.int32(wvn_np.size),
                np.float64(molarMass),
                np.int32(offset))
            offset+=magic_num
        # abs_np = np.empty_like(wvn_np)
        cl.enqueue_copy(self.queue, out_np, out_g)
        return (wvn_np,out_np)

    def VoigtCMPSim( self, 
            featureDatabase : np.array,
            wvn_np: float,
            wvnStart : float,
            wvnStep : float,
            ):
    
        magic_numbers = [2000,6000,8000,16000,32000,64000] #for some reason, running in batches of this many lines at a time is orders of magnitude faster 
        magic_num = 2000

        wvn_np = wvn_np.astype(np.float64)
        # wvn_np  = np.arange(wvnStart,wvnEnd,wvnStep).astype(np.float64)

        wvn_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, wvn_np.nbytes, hostbuf = wvn_np) #wvns
        
         # Allocate complex output buffer
        out_np = np.zeros(wvn_np.size, dtype=np.complex128)
        
        out_g = cl.Buffer(self.ctx,
                      self.mf.WRITE_ONLY,
                      out_np.nbytes)

        # abs_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, wvn_np.nbytes) #absorbance

        zero16 = np.zeros(1, dtype=np.complex128)  # 16 bytes = 128 bits
        cl.enqueue_fill_buffer(self.queue, out_g, zero16.tobytes(), 0, out_np.nbytes)

        db_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, featureDatabase.nbytes, hostbuf = featureDatabase) #feature DB

        knl = self.knl_lineshapeVoigt

        offset = 0
        numFeats = featureDatabase.size
        while(offset<=np.size(featureDatabase)):
            nleft = numFeats-offset
            #find magic number
            magic_num = magic_numbers[0]
            if(nleft > magic_numbers[-1]):
                magic_num = magic_numbers[-1]
            else:
                for i in range(len(magic_numbers)-1):
                    if(nleft >= magic_numbers[i] and nleft <= magic_numbers[i+1]):
                        magic_num = magic_numbers[i]

            kernelSize = min(nleft,magic_num)
            if(kernelSize == 0):
                break
            knl(self.queue,(kernelSize,1),None,
                wvn_g,
                db_g,
                out_g,
                np.float64(wvnStart),
                np.float64(wvnStep),
                np.int32(wvn_np.size),
                np.int32(offset))
            
            offset+=magic_num
        # abs_np = np.empty_like(wvn_np)
        cl.enqueue_copy(self.queue, out_np, out_g)
        return (wvn_np,out_np)

    #maps molecule id to molar mass and isotope abundance
    molMassMap = {"H2O1" : 18.010565 ,  "H2O2" : 20.014811 ,  "H2O3" : 19.01478 ,  "H2O4" : 19.01674 ,  "H2O5" : 21.020985 ,  "H2O6" : 20.020956 ,  "H2O7" : 20.022915 ,  "CO21" : 43.98983 ,  "CO22" : 44.993185 ,  "CO23" : 45.994076 ,  "CO24" : 44.994045 ,  "CO25" : 46.997431 ,  "CO26" : 45.9974 ,  "CO27" : 47.99832 ,  "CO28" : 46.998291 ,  "CO29" : 45.998262 ,  "CO210" : 49.001675 ,  "CO211" : 48.001646 ,  "CO212" : 47.001618 ,  "O31" : 47.984745 ,  "O32" : 49.988991 ,  "O33" : 49.988991 ,  "O34" : 48.98896 ,  "O35" : 48.98896 ,  "N2O1" : 44.001062 ,  "N2O2" : 44.998096 ,  "N2O3" : 44.998096 ,  "N2O4" : 46.005308 ,  "N2O5" : 45.005278 ,  "CO1" : 27.994915 ,  "CO2" : 28.99827 ,  "CO3" : 29.999161 ,  "CO4" : 28.99913 ,  "CO5" : 31.002516 ,  "CO6" : 30.002485 ,  "CH41" : 16.0313 ,  "CH42" : 17.034655 ,  "CH43" : 17.037475 ,  "CH44" : 18.04083 ,  "O21" : 31.98983 ,  "O22" : 33.994076 ,  "O23" : 32.994045 ,  "NO1" : 29.997989 ,  "NO2" : 30.995023 ,  "NO3" : 32.002234 ,  "SO21" : 63.961901 ,  "SO22" : 65.957695 ,  "SO23" : 64.961286 ,  "SO24" : 65.966146 ,  "NO21" : 45.992904 ,  "NO22" : 46.989938 ,  "NH31" : 17.026549 ,  "NH32" : 18.023583 ,  "HNO31" : 62.995644 ,  "HNO32" : 63.992678 ,  "OH1" : 17.00274 ,  "OH2" : 19.006986 ,  "OH3" : 18.008915 ,  "HF1" : 20.006229 ,  "HF2" : 21.012404 ,  "HCl1" : 35.976678 ,  "HCl2" : 37.973729 ,  "HCl3" : 36.982853 ,  "HCl4" : 38.979904 ,  "HBr1" : 79.92616 ,  "HBr2" : 81.924115 ,  "HBr3" : 80.932336 ,  "HBr4" : 82.930289 ,  "HI1" : 127.912297 ,  "HI2" : 128.918472 ,  "ClO1" : 50.963768 ,  "ClO2" : 52.960819 ,  "OCS1" : 59.966986 ,  "OCS2" : 61.96278 ,  "OCS3" : 60.970341 ,  "OCS4" : 60.966371 ,  "OCS5" : 61.971231 ,  "OCS6" : 62.966137 ,  "H2CO1" : 30.010565 ,  "H2CO2" : 31.01392 ,  "H2CO3" : 32.014811 ,  "HOCl1" : 51.971593 ,  "HOCl2" : 53.968644 ,  "N21" : 28.006148 ,  "N22" : 29.003182 ,  "HCN1" : 27.010899 ,  "HCN2" : 28.014254 ,  "HCN3" : 28.007933 ,  "CH3Cl1" : 49.992328 ,  "CH3Cl2" : 51.989379 ,  "H2O21" : 34.00548 ,  "C2H21" : 26.01565 ,  "C2H22" : 27.019005 ,  "C2H23" : 27.021825 ,  "C2H61" : 30.04695 ,  "C2H62" : 31.050305 ,  "PH31" : 33.997241 ,  "COF21" : 65.991722 ,  "COF22" : 66.995078 ,  "SF61" : 145.962494 ,  "H2S1" : 33.987721 ,  "H2S2" : 35.983515 ,  "H2S3" : 34.987105 ,  "HCOOH1" : 46.00548 ,  "HO21" : 32.997655 ,  "O1" : 15.994915 ,  "ClONO21" : 96.956672 ,  "ClONO22" : 98.953723 ,  "NO+1" : 29.997989 ,  "HOBr1" : 95.921076 ,  "HOBr2" : 97.91903 ,  "C2H41" : 28.0313 ,  "C2H42" : 29.034655 ,  "CH3OH1" : 32.026215 ,  "CH3Br1" : 93.941811 ,  "CH3Br2" : 95.939764 ,  "CH3CN1" : 41.026549 ,  "CF41" : 87.993616 ,  "C4H21" : 50.01565 ,  "HC3N1" : 51.010899 ,  "H21" : 2.01565 ,  "H22" : 3.021825 ,  "CS1" : 43.97207 ,  "CS2" : 45.967866 ,  "CS3" : 44.975425 ,  "CS4" : 44.971456 ,  "SO31" : 79.956815 ,  "C2N21" : 52.006148 ,  "COCl21" : 97.93262 ,  "COCl22" : 99.929672 ,  "SO1" : 47.966986 ,  "SO2" : 49.962782 ,  "SO3" : 49.971231 ,  "CH3F1" : 34.021878 ,  "GeH41" : 77.952479 ,  "GeH42" : 75.95338 ,  "GeH43" : 73.95555 ,  "GeH44" : 76.954764 ,  "GeH45" : 79.952703 ,  "CS21" : 75.94414 ,  "CS22" : 77.939936 ,  "CS23" : 76.943526 ,  "CS24" : 76.947495 ,  "CH3I1" : 141.927947 ,  "NF31" : 70.998286}
    isoAbundanceMap = {"H2O1" : 0.997317 ,  "H2O2" : 0.00199983 ,  "H2O3" : 0.000371884 ,  "H2O4" : 0.000310693 ,  "H2O5" : 6.23003e-07 ,  "H2O6" : 1.15853e-07 ,  "H2O7" : 2.41974e-08 ,  "CO21" : 0.984204 ,  "CO22" : 0.0110574 ,  "CO23" : 0.00394707 ,  "CO24" : 0.000733989 ,  "CO25" : 4.43446e-05 ,  "CO26" : 8.24623e-06 ,  "CO27" : 3.95734e-06 ,  "CO28" : 1.4718e-06 ,  "CO29" : 1.36847e-07 ,  "CO210" : 4.446e-08 ,  "CO211" : 1.65354e-08 ,  "CO212" : 1.53745e-09 ,  "O31" : 0.992901 ,  "O32" : 0.00398194 ,  "O33" : 0.00199097 ,  "O34" : 0.000740475 ,  "O35" : 0.000370237 ,  "N2O1" : 0.990333 ,  "N2O2" : 0.00364093 ,  "N2O3" : 0.00364093 ,  "N2O4" : 0.00198582 ,  "N2O5" : 0.00036928 ,  "CO1" : 0.986544 ,  "CO2" : 0.0110836 ,  "CO3" : 0.00197822 ,  "CO4" : 0.000367867 ,  "CO5" : 2.2225e-05 ,  "CO6" : 4.13292e-06 ,  "CH41" : 0.988274 ,  "CH42" : 0.0111031 ,  "CH43" : 0.000615751 ,  "CH44" : 6.91785e-06 ,  "O21" : 0.995262 ,  "O22" : 0.00399141 ,  "O23" : 0.000742235 ,  "NO1" : 0.993974 ,  "NO2" : 0.00365431 ,  "NO3" : 0.00199312 ,  "SO21" : 0.945678 ,  "SO22" : 0.0419503 ,  "SO23" : 0.00746446 ,  "SO24" : 0.00379256 ,  "NO21" : 0.991616 ,  "NO22" : 0.00364564 ,  "NH31" : 0.995872 ,  "NH32" : 0.00366129 ,  "HNO31" : 0.98911 ,  "HNO32" : 0.00363643 ,  "OH1" : 0.997473 ,  "OH2" : 0.00200014 ,  "OH3" : 0.000155371 ,  "HF1" : 0.999844 ,  "HF2" : 0.000155741 ,  "HCl1" : 0.757587 ,  "HCl2" : 0.242257 ,  "HCl3" : 0.000118005 ,  "HCl4" : 3.7735e-05 ,  "HBr1" : 0.506781 ,  "HBr2" : 0.493063 ,  "HBr3" : 7.89384e-05 ,  "HBr4" : 7.68016e-05 ,  "HI1" : 0.999844 ,  "HI2" : 0.000155741 ,  "ClO1" : 0.755908 ,  "ClO2" : 0.24172 ,  "OCS1" : 0.937395 ,  "OCS2" : 0.0415828 ,  "OCS3" : 0.0105315 ,  "OCS4" : 0.00739908 ,  "OCS5" : 0.00187967 ,  "OCS6" : 0.000467176 ,  "H2CO1" : 0.986237 ,  "H2CO2" : 0.0110802 ,  "H2CO3" : 0.00197761 ,  "HOCl1" : 0.75579 ,  "HOCl2" : 0.241683 ,  "N21" : 0.992687 ,  "N22" : 0.00729916 ,  "HCN1" : 0.985114 ,  "HCN2" : 0.0110676 ,  "HCN3" : 0.00362174 ,  "CH3Cl1" : 0.748937 ,  "CH3Cl2" : 0.239491 ,  "H2O21" : 0.994952 ,  "C2H21" : 0.977599 ,  "C2H22" : 0.0219663 ,  "C2H23" : 0.00030455 ,  "C2H61" : 0.97699 ,  "C2H62" : 0.0219526 ,  "PH31" : 0.999533 ,  "COF21" : 0.986544 ,  "COF22" : 0.0110837 ,  "SF61" : 0.95018 ,  "H2S1" : 0.949884 ,  "H2S2" : 0.0421369 ,  "H2S3" : 0.00749766 ,  "HCOOH1" : 0.983898 ,  "HO21" : 0.995107 ,  "O1" : 0.997628 ,  "ClONO21" : 0.74957 ,  "ClONO22" : 0.239694 ,  "NO+1" : 0.993974 ,  "HOBr1" : 0.505579 ,  "HOBr2" : 0.491894 ,  "C2H41" : 0.977294 ,  "C2H42" : 0.0219595 ,  "CH3OH1" : 0.98593 ,  "CH3Br1" : 0.500995 ,  "CH3Br2" : 0.487433 ,  "CH3CN1" : 0.973866 ,  "CF41" : 0.98889 ,  "C4H21" : 0.955998 ,  "HC3N1" : 0.963346 ,  "H21" : 0.999688 ,  "H22" : 0.000311432 ,  "CS1" : 0.939624 ,  "CS2" : 0.0416817 ,  "CS3" : 0.0105565 ,  "CS4" : 0.00741668 ,  "SO31" : 0.943434 ,  "C2N21" : 0.970752 ,  "COCl21" : 0.566392 ,  "COCl22" : 0.362235 ,  "SO1" : 0.947926 ,  "SO2" : 0.04205 ,  "SO3" : 0.00190079 ,  "CH3F1" : 0.988428 ,  "GeH41" : 0.365172 ,  "GeH42" : 0.274129 ,  "GeH43" : 0.205072 ,  "GeH44" : 0.0775517 ,  "GeH45" : 0.0775517 ,  "CS21" : 0.892811 ,  "CS22" : 0.0792103 ,  "CS23" : 0.0140944 ,  "CS24" : 0.0100306 ,  "CH3I1" : 0.988428 ,  "NF31" : 0.996337}