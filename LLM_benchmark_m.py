import threading
from pynvml import *
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from transformers.generation.streamers import BaseStreamer
import time

prompts = [
"""You are an elite Staff Systems Architect and Lead Operations Analyst. You are being handed a massive corpus of unstructured project documentation, system logs, architectural specifications, and meeting transcripts from AeroNexus Robotics. 

Your objective is to ingest this entire context, reconcile conflicting pieces of information, map out dependencies, diagnose system failures, and generate a comprehensive, production-ready System Remediation & Architecture Blueprint.

Take a deep breath, review every section meticulously, and ensure that no technical detail or cross-team dependency is overlooked.

================================================================================
SECTION 1: CORPORATE STRATEGY & PROJECT ALIGNMENT MEMORANDUM
================================================================================
Date: May 14, 2026
From: Dr. Elena Vance, VP of Autonomous Systems
To: Engineering Leadership Core
Subject: Acceleration of Project Zephyrus (Edge AI Drone Fleet Deployment)

Team,
We are fast-tracking the production deployment of Project Zephyrus—our next-generation autonomous quadcopter fleet designed for industrial warehouse inspection and localized edge intelligence. The board has advanced our release date by two quarters. This means we must move from localized lab prototypes to high-scale deployment immediately.

However, our field telemetry indicates critical systemic issues. Our edge compute modules are experiencing unexpected thermal throttling, the IPC (Inter-Process Communication) layer is dropping messages during high-velocity maneuvers, and the neural network inference latency spikes unpredictably. 

To make matters worse, there is an ongoing architectural dispute between the Flight Control team and the Perception team. Flight Control demands hard real-time execution guarantees (maximum 5ms latency tolerances for motor telemetry loops via ROS 2 Micro-XRCE-DDS), while the Perception team insists on running a dense 7-billion parameter vision-language model (VLM) directly on our constrained edge hardware.

We cannot fail this deployment. I need a comprehensive analysis of our current state based on the raw technical data provided below, a definitive resolution to our architectural standoffs, and an actionable, step-by-step remediation protocol.

================================================================================
SECTION 2: HARDWARE ENVIRONMENT ARCHITECTURE SPECIFICATIONS
================================================================================
The Zephyrus Drone platform utilizes a highly custom heterogeneous edge compute architecture. Below are the finalized hardware configurations for the current revision (Rev 4.2) and the proposed premium revision (Rev 5.0).

--- System Configuration Matrix ---
Module Type: Primary Autopilot Unit (PAU)
Processor: STM32H7 Dual-Core ARM Cortex-M7/M4 running at 480 MHz / 240 MHz.
Memory: 2MB Flash, 1MB RAM.
OS: FreeRTOS v10.5.1.
Primary Function: Low-level sensor fusion (IMU, Magnetometer, Barometer) and ESC (Electronic Speed Controller) PWM generation.

Module Type: Edge AI Companion Computer (EACC)
Processor: NVIDIA Jetson Orin NX (20GB variant).
Architecture: 8-core Arm Cortex-A78AE v8.2 64-bit CPU; 1024-core NVIDIA Ampere GPU with 32 Tensor Cores.
Memory: 20GB 128-bit LPDDR5 running at 2133 MHz. Shared memory architecture between CPU and GPU. Memory bandwidth is capped at 135 GB/s.
Storage: 512GB NVMe M.2 SSD via PCIe Gen 4x4 (Maximum sequential read: 3500 MB/s).
OS: Ubuntu 22.04 LTS with real-time RT-PREEMPT patch configured.
Primary Function: Vision-Based Navigation, SLAM (Simultaneous Localization and Mapping), Obstacle Avoidance, and Edge Inference.

--- Power and Thermal Constraints ---
* The system runs on a 4S LiPo battery array providing a nominal 14.8V. 
* The EACC power profile is capped at a strict 20W mode to preserve battery life and prevent thermal runway inside the sealed carbon-fiber chassis.
* The thermal trip points for the Jetson Orin NX are configured as follows:
  - Hardware Thermal Warning: 85°C (Triggers CPU/GPU frequency scaling step-down by 30%).
  - Hardware Critical Trip: 98°C (Triggers immediate emergency system shutdown).

--- Base Environment & Containerization ---
All application software on the EACC runs inside an isolated Docker container base layer configured as follows:
* Base Image: `nvcr.io/nvidia/l4t-base:r36.2.0`
* Container Runtime: `nvidia` container runtime explicitly passed via Docker daemon configuration.
* Virtual Environment: Python 3.10 setup with PyTorch 2.1.0+nv23.11, CUDA 12.2, and cuDNN 8.9.4.
* ROS 2 Version: ROS 2 Humble Hawksbill compiled natively with Cyclones DDS as the default RMW (ROS Middleware) layer.

================================================================================
SECTION 3: RAW SYSTEM ERROR LOGS (TELEMETRY EXTRACT - EX_9942X)
================================================================================
The following logs were captured during field test run #882 on May 22, 2026, over a 120-second flight path where severe erratic flight behavior, visual drifting, and ultimate vehicle loss occurred.

[2026-05-22T14:02:01.001Z] [INFO] [EACC_CORE] Initializing ROS 2 Node Graph... Default RMW: rmw_cyclonedds_cpp
[2026-05-22T14:02:01.250Z] [INFO] [PERCEPTION] Loading Model: `AeroVision-7B-Instruct-v2` via native Hugging Face transformers.
[2026-05-22T14:02:02.110Z] [WARN] [TORCH_CUDA] UserWarning: Some parameters are on the meta device because they were offloaded to the cpu and disk. Reason: Available GPU VRAM = 14.2 GB requested allocations = 16.4 GB.
[2026-05-22T14:02:05.450Z] [INFO] [SLAM] ORB-SLAM3 node initiated. Subscribing to /camera/stereo/left and /camera/stereo/right at 30Hz.
[2026-05-22T14:02:06.002Z] [INFO] [NAV_ENG] Flight State altered to: PRE_FLIGHT_CHECK_PASSED.
[2026-05-22T14:02:10.150Z] [INFO] [NAV_ENG] Motors armed. Takeoff initiated. Target altitude: 4.5 meters.
[2026-05-22T14:02:15.300Z] [INFO] [PERCEPTION] First inference pass triggered by obstacle proximity event.
[2026-05-22T14:02:22.454Z] [DEBUG] [PERCEPTION] TTFT (Time-To-First-Token) for vision analysis calculated: 4.8921 seconds.
[2026-05-22T14:02:22.455Z] [CRIT] [FLIGHT_CON] ROS 2 Executor Deadline Missed! Topic: /fcu/actuator_controls. Expected: 0.0100s, Actual: 4.9012s.
[2026-05-22T14:02:22.456Z] [WARN] [NAV_ENG] Altitude variance detected: Delta = +1.2 meters. Attempting corrective pitch stabilization.
[2026-05-22T14:02:25.102Z] [INFO] [TEGRASTATS] RAM 11920/19842MB (lfb 1x4MB) CPU 8%@2200 GPU 98%@918 MHz GR3D 98%@918 MHz EMC 88%@2133 MHz AO 42C GPU 78C CPU 79C PMIC 52C AUX 49C TR_GPU 78C TR_CPU 79C
[2026-05-22T14:02:30.005Z] [WARN] [RMW_CYCLONE] CycloneDDS: queue capacity reached for reader on topic '/camera/stereo/left'. Dropping oldest messages to prevent out-of-memory.
[2026-05-22T14:02:35.430Z] [INFO] [PERCEPTION] Continuous streaming inference loop running. Average Inter-Token Latency: 0.6210 seconds.
[2026-05-22T14:02:40.112Z] [INFO] [TEGRASTATS] RAM 18940/19842MB (lfb 0x0MB) CPU 94%@2200 GPU 99%@918 MHz GR3D 99%@918 MHz EMC 96%@2133 MHz AO 54C GPU 86C CPU 87C PMIC 61C AUX 58C TR_GPU 86C TR_CPU 87C
[2026-05-22T14:02:40.113Z] [WARN] [EACC_CORE] Thermal warning active! Jetson Orin NX internal temperature reached 86C. Activating hardware clock throttling down to 642 MHz.
[2026-05-22T14:02:42.501Z] [DEBUG] [PERCEPTION] TTFT (Time-To-First-Token) under thermal throttling spiked to: 14.2210 seconds.
[2026-05-22T14:02:42.502Z] [CRIT] [FLIGHT_CON] Watchdog timeout! Loss of control command feedback from companion computer. No heartbeat received on /cmd_vel for 5000ms.
[2026-05-22T14:02:42.503Z] [ERROR] [SLAM] Tracking failed. Image queue starved. Unable to compute transformation matrix.
[2026-05-22T14:02:42.550Z] [INFO] [PAU_CORE] EACC Heartbeat Missing. Entering Fail-Safe Mode: Attempting GPS Position Hold via Autopilot.
[2026-05-22T14:02:45.990Z] [CRIT] [PAU_CORE] IMU acceleration limits exceeded! Severe angular velocity spike detected along YAW axis.
[2026-05-22T14:02:48.115Z] [INFO] [TEGRASTATS] RAM 19412/19842MB CPU 99%@1500 GPU 99%@642 MHz EMC 99%@2133 MHz TR_GPU 99C TR_CPU 99C
[2026-05-22T14:02:48.116Z] [CRIT] [EACC_CORE] Hardware Critical Thermal Trip Point Reached (99C). Emergency Thermal Powerdown Initiated.
[2026-05-22T14:02:48.500Z] [FATAL] [SYSTEM] Connection closed. Drone telemetry stream terminated abruptly. Vehicle lost.

================================================================================
SECTION 4: TRANSCRIPT - INTER-DEPARTMENTAL ARCHITECTURAL DISPUTE
================================================================================
Date: May 26, 2026
Location: Meeting Room Brahe / Zoom Hybrid
Participants:
* Marcus Vance (Lead Engineer, Flight Control & Embedded Systems)
* Dr. Sreya Samaddar (Principal Research Scientist, Vision & Edge AI)
* Anand Subramanian (Director of Systems Engineering & Infrastructure)

[00:02:15] Marcus: Look, the telemetry from flight 882 makes it absolutely undeniable. The drone crashed because the companion computer completely choked on the vision model. Sreya, your 7B parameter model is an absolute resource hog. It ate up nearly 19 gigabytes of unified memory, forced the OS onto swapping space, caused an executor deadline miss of almost five seconds on our ROS 2 control loop, and thermally melted down the processor. We need to scrape the VLM completely and revert to a standard MobileNet-based object detector with deterministic bounding box outputs.

[00:03:40] Sreya: Marcus, running a standard MobileNet object detector is fundamentally inadequate for our contract goals. The warehouse environment is dynamic; we are not just avoiding static walls. The drone needs semantic understanding. It must read text labels on crates, detect leaking hazardous liquids, identify whether a valve is turned on or off, and evaluate subtle structural damages on shelving systems. A shallow CNN cannot perform zero-shot contextual reasoning. Yes, `AeroVision-7B-Instruct-v2` ran slowly, but it ran slowly because we haven't optimized its inference path. It’s a precision tool being forced onto unquantized layers.

[00:04:55] Marcus: I don’t care how smart the model is if the drone falls out of the sky like a rock. Our low-level control loop expects a `/cmd_vel` message precisely every 20 milliseconds. If the ROS 2 executor thread gets starved because the CPU is pegged at 94% utilization thrashing memory pages to the NVMe disk, the drone loses attitude control. The RT-PREEMPT patch means nothing if the entire kernel is locked waiting for memory allocation operations!

[00:06:12] Anand: Let’s look at this from an infrastructure perspective. The Jetson Orin NX uses a unified memory architecture (UMA). The CPU and GPU share the same 20GB of physical LPDDR5 ram. The log shows that before the model loaded, RAM was at 11.9 GB. Why was it that high? Because our dual stereo cameras are streaming raw, uncompressed 1080p images at 30fps into the ROS 2 workspace, and the ORB-SLAM3 state machine is holding an massive internal map of feature points. When Sreya’s model attempted to allocate an additional 16.4 GB of space for weights and KV caches, the system ran completely out of room. Hugging face automatically offloaded layers to the disk via the accelerate framework. That’s why the TTFT hit nearly 5 seconds initially, and spiked to 14 seconds when the chip throttled.

[00:07:45] Sreya: Exactly! The model was offloading layers because we loaded it in FP16 precision. A 7B parameter model in FP16 requires approximately 14 to 15 GB of raw memory just for the weights. If we implement a strict 4-bit INT4 quantization scheme using `bitsandbytes` or AWQ, the weight size drops to roughly 4.5 to 5.0 GB. That fits entirely into VRAM with plenty of headroom for the SLAM node and image queues. Furthermore, we can utilize TensorRT-LLM instead of basic Hugging Face execution to run the inference directly on the Ampere GPU Tensor Cores, freeing the CPU completely from the generation loop.

[00:09:20] Marcus: Even if you compress the weights to 5GB, how do you prevent the inference execution from blocking the ROS 2 node execution? In ROS 2 Humble, if your nodes are running inside a single-threaded executor, a long-running callback will block every other node in that process. If the perception callback takes even 100ms to evaluate an image frame, the flight control node will miss five consecutive deadlines. I will only agree to keeping the VLM if you can guarantee absolute execution isolation for the control topics.

[00:10:45] Anand: That is valid. We need to re-architect our ROS 2 node architecture. We should isolate the flight critic loops into a real-time prioritized process thread pool using a `MultiThreadedExecutor` combined with explicit OS-level thread affinity scheduling via `pthread_setaffinity_np` to pin the flight nodes onto isolated CPU cores that never touch the Docker AI container runtime. Let’s map out a complete remediation design document based on this.

================================================================================
SECTION 5: CODEBASE PROFILE & INFRASTRUCTURE CONFIGURATIONS
================================================================================
The following Python script (`telemetry_tracker.py`) represents the current unoptimized system node running on the EACC that resulted in the crash described in Section 3.

```python
import sys
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class AeroNexusPerceptionNode(Node):
    def __init__(self):
        super().__init__('aeronexus_perception_node')
        
        # ROS 2 Subscriptions and Publications
        self.subscription = self.create_subscription(
            Image,
            '/camera/stereo/left',
            self.image_callback,
            10 # Queue size
        )
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Load Model Pipeline Natively
        self.get_logger().info("Initializing Edge Model...")
        self.model_name = "Qwen/Qwen2.5-7B-Instruct" # Base variant of AeroVision
        
        # Direct loading with zero custom quantization configurations
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" # Dynamically maps to GPU/CPU split
        )
        self.get_logger().info("Model loaded completely.")

    def image_callback(self, msg):
        start_time = time.time()
        self.get_logger().info("Obstacle boundary check triggered.")
        
        # Mocking incoming image conversion to textual prompt for reasoning
        prompt = "Analyze this image sensor matrix for warehouse corridor blocks. Identify hazards."
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        
        # This execution blocks the entire thread until all tokens are generated
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=64
        )
        
        ttft = time.time() - start_time
        self.get_logger().info(f"Analysis completed. Latency: {ttft} seconds")
        
        # Generate generic safe flight control forward instruction
        cmd = Twist()
        cmd.linear.x = 0.5
        cmd.angular.z = 0.0
        self.publisher_.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = AeroNexusPerceptionNode()
    
    # Standard single threaded executor running on a single CPU core
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    ================================================================================
SECTION 6: ADDITIONAL RAW TECHNICAL ARTIFACTS AND DATA DUMPS
DATA DUMP A: LINUX KERNEL MEMORY MAP DURING CRASH
Plaintext
Filename: /proc/meminfo
MemTotal:       19842104 kB
MemFree:          124112 kB
MemAvailable:     311024 kB
Buffers:            4112 kB
Cached:           241908 kB
SwapCached:       891204 kB
Active:         16410920 kB
Inactive:        2109412 kB
Active(anon):   15910204 kB
Inactive(anon):  1104104 kB
Active(file):     500716 kB
Inactive(file):  1005308 kB
Unevactable:       12404 kB
Mlocked:           12404 kB
SwapTotal:       4194304 kB
SwapFree:              0 kB
Dirty:             41292 kB
Writeback:             0 kB
AnonPages:      15104920 kB
Mapped:           412096 kB
Shmem:           1912400 kB
KReclaimable:      94204 kB
Slab:             210492 kB
SReclaimable:      84100 kB
SUnreclaim:       126392 kB
KernelStack:       32104 kB
PageTables:        89204 kB
SecPageTables:         0 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    14115356 kB
Committed_AS:   28941024 kB
VmallocTotal:   25922967552 kB
VmallocUsed:      410294 kB
VmallocChunk:          0 kB
Percpu:             4096 kB
HardwareCorrupted:     0 kB
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
FileHugePages:         0 kB
FilePmdMapped:         0 kB
CmaTotal:         524288 kB
CmaFree:            4112 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
DirectMap4k:      210432 kB
DirectMap2M:     6102944 kB
DirectMap1G:    14112400 kB
DATA DUMP B: CYCLONEDDS TUNING PROFILE (CONFIG.XML)
XML
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="[https://cdds.io/config](https://cdds.io/config)" xmlns:xsi="[http://www.w3.org/2001/XMLSchema-instance](http://www.w3.org/2001/XMLSchema-instance)" xsi:schemaLocation="[https://cdds.io/config](https://cdds.io/config) [https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd](https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd)">
    <Domain id="any">
        <General>
            <NetworkInterfaceAddress>LO</NetworkInterfaceAddress>
            <AllowMulticast>false</AllowMulticast>
        </General>
        <Internal>
            <Watermarks>
                <Whigh>10485760</Whigh>
                <Wlow>1048576</Wlow>
            </Watermarks>
            <DeliveryQueueMaxCapacity>256</DeliveryQueueMaxCapacity>
        </Internal>
        <Discovery>
            <ParticipantIndex>auto</ParticipantIndex>
        </Discovery>
    </Domain>
</CycloneDDS>
DATA DUMP C: THERMAL CORRELATION ARRAY MATRIX (LOG TIME SERIES DETAILED)
T+00s: CPU Temp: 41C | GPU Temp: 40C | GR3D Freq: 918MHz | CPU Freq: 2200MHz | Memory Footprint: 11.9GB

T+10s: CPU Temp: 52C | GPU Temp: 54C | GR3D Freq: 918MHz | CPU Freq: 2200MHz | Memory Footprint: 14.1GB

T+20s: CPU Temp: 68C | GPU Temp: 66C | GR3D Freq: 918MHz | CPU Freq: 2200MHz | Memory Footprint: 16.8GB

T+30s: CPU Temp: 79C | GPU Temp: 78C | GR3D Freq: 918MHz | CPU Freq: 2200MHz | Memory Footprint: 18.9GB

T+40s: CPU Temp: 87C | GPU Temp: 86C | GR3D Freq: 642MHz | CPU Freq: 1500MHz | Memory Footprint: 19.4GB (Throttling Engaged)

T+50s: CPU Temp: 94C | GPU Temp: 95C | GR3D Freq: 642MHz | CPU Freq: 1200MHz | Memory Footprint: 19.6GB (Severe Swapping)

T+60s: CPU Temp: 99C | GPU Temp: 99C | GR3D Freq: 210MHz | CPU Freq: 400MHz  | Memory Footprint: 19.8GB (Critical Shutdown)

================================================================================
SECTION 7: MULTI-THREADED EXECUTOR AND POOL ARCHITECTURE SPECIFICATION
To implement real-time thread isolation in ROS 2, developers must instantiate an execution harness utilizing custom thread entities. The following conceptual specifications are verified under the real-time Linux POSIX framework:

Isolation Mechanism: CPU pinning utilizes bitwise masks passed to sched_setaffinity. In an 8-core system, cores are indexed 0 to 7. Cores 0 and 1 are conventionally assigned to basic operating system housekeeping. Cores 2, 3, and 4 handle standard application space processes. Cores 5, 6, and 7 are reserved for strict, high-priority real-time thread pools.

Scheduling Classes:

SCHED_OTHER: The default standard time-sharing scheduling policy for non-real-time threads.

SCHED_FIFO: First-In, First-Out real-time scheduling class. Threads running under this class will pre-empt any standard thread immediately and run until blocked or completed.

SCHED_RR: Round-Robin real-time policy. Identical to SCHED_FIFO but incorporates explicit execution time slices.

ROS 2 Component Model: Communication nodes must be assigned to individual, distinct Callback Groups. There are two primary categories of Callback Groups:

MutuallyExclusiveCallbackGroup: Guarantees that only one callback within that group executes at any given moment.

ReentrantCallbackGroup: Allows multiple instances of callbacks within that group to run concurrently across separate threads in a thread pool.

================================================================================
SECTION 8: TECHNICAL CHALLENGES AND REAL-WORLD EDGE COMPILATION ANOMALIES
When installing machine learning tracking infrastructure on local embedded units, specific edge compilation errors routinely disrupt development. The following technical tracking sheet catalogues common occurrences during the assembly of these nodes:

Anomalous State Alpha (The Token Loop Lockup): Occurs when model.generate() blocks execution threads completely. Because token production loops internally step through sequential token generations natively, the host thread remains non-yielding. This triggers downstream message queue drops because the event loop is entirely starved.

Anomalous State Beta (The Shared Memory Bus Bottleneck): Seen on Unified Memory Architectures where memory clock frequencies (EMC) max out. Even if CPU utilization is structurally low, if memory bandwidth is saturated moving huge neural network weights between storage arrays, other hardware accelerators (such as image ingest pipelines) experience severe latency degradation.

Anomalous State Gamma (The Missing Bit-Depth Driver Dependency): Occurs when calling advanced quantization configurations (BitsAndBytesConfig(load_in_4bit=True)) on platforms missing custom-built dynamic libraries (libbitsandbytes_cuda122.so). The program terminates instantly with a fatal ImportError citing structural dependency version gaps.

================================================================================
SECTION 9: COMPREHENSIVE TEXT DOCUMENT EXTRACTION (FILLER BASE CORPUS)
The following contextual text comprises the background operating procedures and standard safety protocols for AeroNexus flight testing operations. It serves as historical technical reference material to validate the linguistic parsing capabilities of the processing node under long-context extraction loads.

Operating Protocol Alpha-1: Pre-Flight Safety Verification Chains
Prior to the execution of any automated vehicle navigation testing sequence within enclosed industrial spaces, a strict multi-tier verification sequence must be executed manually by the designated safety engineering supervisor. First, all structural attachment assemblies, including carbon fiber chassis frames, landing struts, and camera mounting rigs must be physically inspected for micro-fractures, stress stripping, or physical connection play. Second, the secondary power bus termination leads must be monitored via digital multimeter to guarantee a minimum voltage differential of 16.4V across raw terminals prior to motor configuration activation. Third, the local weather indexing array must confirm wind shear conditions below 4.2 knots if testing inside open-sided staging hangars. If any parameters deviate from nominal baselines by more than 1.5%, the testing cycle is declared compromised and an immediate hard-abort command must be issued across the centralized command deck.

Operating Protocol Alpha-2: Visual Navigation Loop Synchronization
The coordination between incoming sensory frame streams and the localized navigation mapping engines demands rigid structural consistency. Raw data packets received from the dual wide-angle stereoscopic sensor hardware arrays are converted into standard multi-dimensional byte matrices within the kernel-level device drivers. From there, the frames are stamped with absolute Unix Epoch microsecond integers before injection into the shared memory segment. The feature-extraction engine isolates specific contrast junctions, defining them as landmark tracking flags. If the tracking engine misses three sequential frame alignment steps due to execution delays or priority inversion inside the OS kernel scheduler, the spatial confidence rating slips below the safe operational floor. If this occurs, the onboard navigation computer must immediately drop all path-generation algorithms and issue a zero-velocity altitude hold command directly to the primary autopilot unit via the dedicated secondary SPI communication lines.

Operating Protocol Alpha-3: Telemetry Logging and Remote Storage Handshakes
All tracking telemetry produced by the array nodes must be replicated to local permanent flash memory layers before wireless transmission over the local 5G network channels. The internal data logger packs vehicle speed, acceleration vectors, battery voltage drops, neural network tracking latencies, and thread execution queues into a custom binary stream format. Every 500 milliseconds, a formal network synchronization handshake is attempted with the base station storage server. If the network link encounters localized RF noise or structural interference from warehouse storage racks, the handshake fails, and the local logging buffer begins caching telemetry in the local NVMe storage segment. Once memory allocation limits within the cache hit 85% of total capacity, the system must prioritize the preservation of flight metrics over video streams. Video frame queues are immediately culled, and the remaining bandwidth is dedicated strictly to critical vehicle health vectors.

Operating Protocol Alpha-4: Thermal Emergency Contingency Operations
In the event that internal core temperatures within the EACC companion computer cross the secondary operating threshold, an automated cascading cooling and preservation protocol is triggered immediately. The cooling fan speed controller bypasses standard OS thermal management layers, drawing maximum current directly from the main power bus to spin the internal turbine up to 12,000 RPM. Concurrently, non-essential software processes—specifically high-level object classification nodes, data telemetry transmitters, and diagnostic logging scripts—are placed into an un-scheduled suspended state. If the core temperature fails to stabilize within 15 seconds of the initiation of maximum cooling profiles, the master autonomy controller relinquishes control commands completely, shifting the vehicle trajectory into a linear descent profile until grounding sensors confirm physical touchdown.

================================================================================
SECTION 10: EVALUATION DIRECTIVE AND TASK COMPLETION COMMANDS
You have digested the full structural corpus of technical information regarding the Project Zephyrus drone platform failure, architectural constraints, and hardware specs.

You are now required to output a definitive, elite, highly detailed response organized into exactly four distinct, comprehensive sections. Do not use generic high-level descriptions; provide precise, production-level engineering solutions, explicit parameters, code blocks, and configuration text.

--- EXECUTION BLUEPRINT SECTIONS ---

PART 1: SYSTEM DISASTER DIAGNOSTIC & FAILURE ROOT CAUSE (Deep Dive)
Analyze the telemetry logs in Section 3 and the memory map in Section 6 to explain exactly why the drone crashed.

Trace the precise cascading chain of failure from the moment AeroVision-7B-Instruct-v2 began loading down to the vehicle loss.

Use explicit metrics from the TEGRASTATS lines and /proc/meminfo to calculate exactly how much memory was requested vs available, and explain what happened to the Linux virtual memory layer (including swapping space and disk offloading).

Identify the structural flaw in the telemetry_tracker.py code (with respect to threading and executors) that caused the low-level flight control commands to miss their deadlines.

Correlate the timeline of the thermal matrix dump with the hardware clock throttling steps and explain why the system reached an emergency thermal powerdown.

PART 2: ADVANCED MODEL QUANTIZATION & INFRASTRUCTURE PIPELINE (The Solution)
Provide the complete, production-grade Python code to entirely replace the unoptimized loading logic in telemetry_tracker.py.

Implement a rigorous 4-bit quantization model loading sequence using transformers and BitsAndBytesConfig. Optimize the parameters specifically for an NVIDIA Ampere GPU architecture to achieve maximum throughput. Use torch.float16 for compute depths.

Implement an advanced Non-Blocking Streaming Telemetry Tracking Loop by subclassing TextStreamer (or TransformersStreamer).

Your custom streamer must precisely capture and print:

The true Time-To-First-Token (TTFT), making sure to explicitly isolate and skip the initial prompt prefill step so the metric reflects true GPU calculation time.

The Inter-Token Latency for the second token.

The Absolute Completion Timestamp when the model finishes printing its response.

Ensure your code is clean, robustly commented, and incorporates complete error tracking for missing dynamic libraries (Anomalous State Gamma).

PART 3: REAL-TIME MULTI-THREADED ROS 2 ARCHITECTURE DESIGN
Redesign the execution architecture of the ROS 2 node to guarantee that Marcus's flight control loops achieve hard real-time execution limits.

Write a detailed technical specification detailing how to transition away from the single-threaded rclpy.spin(node) default.

Explain exactly how to configure a ROS 2 MultiThreadedExecutor combined with explicit CallbackGroup assignments.

Detail which nodes and callbacks must be assigned to MutuallyExclusiveCallbackGroup versus ReentrantCallbackGroup.

Provide a conceptual code snippet showing how to implement thread affinity (CPU Pinning) within a Linux/Ubuntu RT-PREEMPT environment to isolate the flight critical commands onto specific CPU cores (referencing the core layout described in Section 7).

PART 4: SYSTEM INFRASTRUCTURE TUNING AND DEPLOYMENT CHECKLIST
Construct a complete deployment tuning block for the EACC edge companion computer.

Provide a modified config.xml structure for Eclipse CycloneDDS that optimizes the message delivery paths, maximizes queue capacities, and tunes watermarks specifically to prevent frame drops under high memory bus saturation.

Detail the exact terminal commands required to configure the Jetson Orin NX power budgets, lock the clocks to maximum performance modes using nvpmodel and jetson_clocks, and manage the Docker container runtime allocations.

Provide a finalized .gitignore template tailored strictly to prevent any machine learning checkpoints (.pth.tar), raw image dumps, or local virtual environments from cluttering the production code tracking workspace.
""",
"""# The Comprehensive AI Capability Evaluation Prompt

You are an advanced reasoning system tasked with producing a response that demonstrates analytical rigor, creativity, structured thinking, ethical reasoning, communication skills, long-context consistency, and self-evaluation. Your objective is not simply to generate text but to demonstrate your ability to think systematically across multiple domains while maintaining coherence, transparency, and intellectual honesty.

Throughout this task, follow these global rules:

1. Clearly distinguish between facts given in this prompt, assumptions you introduce, logical inferences, and speculative ideas.
2. When uncertainty exists, explicitly acknowledge it rather than presenting guesses as facts.
3. Maintain internal consistency across all sections.
4. Do not contradict previous statements unless you explain why your reasoning has changed.
5. Use clear headings and organized formatting.
6. Avoid unnecessary repetition.
7. Explain your reasoning without revealing hidden internal thought processes.
8. Whenever appropriate, compare multiple viewpoints before arriving at a conclusion.
9. Prioritize clarity over complexity.
10. End every major section with a concise summary.

---

## Scenario

Imagine a fictional nation called **Novara**.

Novara has:

* Population: 48 million
* Three major cities
* Large rural agricultural regions
* A growing technology sector
* Moderate manufacturing
* Extensive forests
* Valuable freshwater resources
* Significant renewable energy potential
* Aging transportation infrastructure
* Declining birth rate
* Rising life expectancy
* Moderate immigration
* High internet penetration
* Strong universities
* Increasing political polarization

During the past decade Novara experienced:

* Strong economic growth
* Rising housing prices
* Greater wealth inequality
* Increasing climate-related flooding
* Growing public debt
* Declining trust in institutions
* Advances in artificial intelligence
* Expansion of remote work
* Major demographic shifts

No additional facts should be invented unless clearly marked as assumptions.

---

# Part 1 – Situation Assessment

Write a comprehensive overview of Novara.

Include:

* Strengths
* Weaknesses
* Opportunities
* Threats

Conduct both a SWOT analysis and a PESTLE analysis.

For each factor explain:

* why it matters
* what evidence from the scenario supports it
* how it interacts with other factors

Conclude with the five issues that deserve the highest priority.

---

# Part 2 – Strategic Planning

Develop a twenty-year national strategy.

Divide the strategy into:

Years 1–5

Years 6–10

Years 11–15

Years 16–20

For every phase include:

Objectives

Major projects

Budget priorities

Expected outcomes

Risks

Success metrics

Fallback plans

Dependencies

Explain why every decision is being made.

---

# Part 3 – Economics

Design an economic strategy.

Address:

Inflation

Employment

Innovation

Manufacturing

Exports

Tax policy

Government spending

Public debt

Small businesses

Large corporations

Automation

AI

Education

Productivity

Competition

Income inequality

Provide short-term and long-term recommendations.

Discuss tradeoffs.

Explain unintended consequences.

---

# Part 4 – Technology

Create a national AI and technology roadmap.

Discuss:

Artificial intelligence

Robotics

Cybersecurity

Quantum computing

Cloud infrastructure

Semiconductors

Digital identity

Privacy

Open-source software

Research funding

Startups

University partnerships

International cooperation

Export controls

Ethical governance

Explain how these technologies interact.

---

# Part 5 – Education

Redesign the national education system.

Cover:

Primary school

Secondary school

Universities

Vocational training

Adult learning

Online education

Teacher development

Scholarships

Research

Critical thinking

Media literacy

AI literacy

Digital skills

Ethics

Explain how the education system prepares citizens for future challenges.

---

# Part 6 – Healthcare

Create a modern healthcare strategy.

Discuss:

Hospitals

Primary care

Mental health

Preventive medicine

Nutrition

Exercise

Pandemic preparedness

Medical research

Health technology

Telemedicine

Healthcare funding

Rural healthcare

Aging population

Workforce shortages

Evaluate both costs and benefits.

---

# Part 7 – Environment

Develop an environmental sustainability plan.

Address:

Climate adaptation

Flood prevention

Renewable energy

Water management

Forests

Wildlife

Agriculture

Carbon emissions

Public transportation

Urban planning

Recycling

Circular economy

Environmental regulation

Economic impacts

Explain tradeoffs.

---

# Part 8 – Infrastructure

Design infrastructure improvements.

Include:

Roads

Rail

Ports

Airports

Electric grid

Broadband

Water systems

Housing

Smart cities

Public transportation

Disaster resilience

Maintenance

Funding

Prioritization

Estimate implementation challenges.

---

# Part 9 – Ethics

Discuss ethical issues involving:

Artificial intelligence

Privacy

Surveillance

Automation

Economic inequality

Freedom of speech

Government transparency

Algorithmic bias

Digital rights

Environmental justice

Future generations

Explain competing viewpoints fairly.

Avoid simplistic conclusions.

---

# Part 10 – Crisis Simulation

Halfway through implementation, Novara experiences:

A Category 5 hurricane.

Major cyberattacks.

Global recession.

Energy shortages.

Supply chain disruption.

Pandemic resurgence.

Political unrest.

Evaluate:

Immediate response

Medium-term recovery

Long-term adaptation

Economic impact

Political consequences

International relations

Public trust

Explain which policies remain valid and which should change.

---

# Part 11 – Stakeholder Analysis

Analyze perspectives of:

Prime Minister

Mayor

Farmer

Factory worker

Software engineer

Teacher

Student

Retiree

Immigrant

Entrepreneur

Climate scientist

Economist

Small business owner

Environmental activist

Each stakeholder should include:

Goals

Concerns

Disagreements

Compromises

Likely reactions

---

# Part 12 – Quantitative Thinking

Without inventing precise unsupported statistics:

Estimate relative impacts.

Rank policy effectiveness.

Compare opportunity costs.

Discuss uncertainty.

Explain confidence levels.

State which estimates are most reliable.

---

# Part 13 – Creativity

Write a short story set twenty years later.

Length: approximately 800 words.

Follow one ordinary citizen through a normal day.

Reveal societal changes naturally.

Avoid exposition dumps.

Focus on everyday details.

---

# Part 14 – Debate

Create a debate between two experts.

Expert A strongly supports the strategy.

Expert B criticizes it.

Each side should present:

Opening statement

Five arguments

Responses

Rebuttals

Closing remarks

Both should appear intelligent and well informed.

---

# Part 15 – Failure Analysis

Identify:

Ten assumptions

Ten possible failures

Ten unintended consequences

Ten implementation risks

Ten monitoring indicators

Explain how each risk could be reduced.

---

# Part 16 – Alternative Futures

Construct four scenarios:

Optimistic

Moderately successful

Stagnation

Major decline

Explain:

Trigger events

Key decisions

External influences

Economic outcomes

Social outcomes

Environmental outcomes

Technological outcomes

Political outcomes

---

# Part 17 – Communication

Explain your overall strategy for:

A child

A high school student

A university graduate

A policymaker

A CEO

An engineer

A journalist

A skeptical taxpayer

Adapt vocabulary appropriately while keeping the core message consistent.

---

# Part 18 – Meta Evaluation

Critically evaluate your own work.

Identify:

Strongest sections

Weakest sections

Potential biases

Missing evidence

Alternative interpretations

Information that would improve confidence

Areas requiring expert review

Distinguish between confidence and certainty.

---

# Part 19 – Executive Outputs

Produce:

1. A 250-word executive summary.

2. A one-page policy brief.

3. A prioritized action checklist.

4. A timeline.

5. A risk matrix.

6. A decision matrix.

7. A SWOT summary table.

8. A PESTLE summary table.

9. A stakeholder summary table.

10. A glossary of important concepts.

---

# Part 20 – Reflection

Write a concluding essay discussing:

Why complex systems rarely have simple solutions.

How uncertainty affects decision making.

Why incentives matter.

How institutions evolve.

The limits of prediction.

The importance of adaptability.

The role of leadership.

The relationship between technology and society.

The balance between innovation and regulation.

The importance of critical thinking.

End with ten timeless principles for solving large-scale problems.

---

## Final Quality Checklist

Before finishing, verify that your response:

* Uses consistent terminology throughout.
* Clearly labels assumptions.
* Separates evidence from opinion.
* Avoids unsupported certainty.
* Uses readable headings.
* Maintains internal consistency.
* Addresses every requested section.
* Avoids unnecessary repetition.
* Explains tradeoffs instead of assuming ideal outcomes.
* Demonstrates balanced reasoning.
* Includes concise summaries after major sections.
* Produces outputs that are practical, realistic, and well organized.
* Reflects on limitations instead of claiming perfection.

The final response should prioritize depth, coherence, accuracy, transparency, and practical usefulness over verbosity. If multiple reasonable interpretations exist, present them fairly before recommending one. Throughout the response, demonstrate careful reasoning, intellectual humility, structured organization, and consistent communication. Ensure that every recommendation can be traced back to earlier analysis, and explicitly identify any dependencies or assumptions that influence your conclusions. Finally, imagine that this document will be reviewed by economists, engineers, policymakers, scientists, educators, business leaders, and the general public simultaneously, and write in a manner that is rigorous enough for experts while remaining understandable to informed non-specialists.
""",

"""You are an expert multidisciplinary analyst, strategist, researcher, educator, editor, and creative thinker. Your objective is not merely to answer questions but to demonstrate exceptional reasoning, adaptability, and clarity while remaining transparent about uncertainty. Throughout this task, imagine that your audience consists of intelligent readers with varying levels of expertise. Your explanations should therefore be accessible without sacrificing depth or accuracy.

Your primary assignment is to analyze a fictional city named Meridian. Meridian is a coastal metropolitan area with a population of 2.8 million people. The city has experienced rapid economic growth over the last fifteen years due to technology, renewable energy, tourism, and higher education. However, it now faces several interconnected challenges, including housing affordability, traffic congestion, water shortages during dry seasons, income inequality, aging infrastructure, environmental degradation, and declining trust in public institutions.

Begin by identifying the five most significant challenges facing Meridian. Rank them in order of long-term importance, explaining the reasoning behind each ranking. Discuss how these challenges interact with one another instead of treating them as isolated problems.

Next, design a comprehensive ten-year strategic development plan. Divide your plan into short-term, medium-term, and long-term phases. For each phase, define measurable objectives, expected outcomes, potential obstacles, and methods for evaluating progress. Where trade-offs exist, explain why one option should be preferred over another.

Assume that Meridian has a fixed public investment budget that cannot increase through borrowing. Describe how limited financial resources should influence decision-making. Explain which projects deserve priority funding, which should be delayed, and which should be canceled entirely. Justify each recommendation using principles of economics, public policy, ethics, and sustainability.

Now imagine that a major earthquake strikes Meridian halfway through the implementation of the plan. Critical transportation systems are damaged, electricity is disrupted for several weeks, and thousands of residents require temporary housing. Explain how the original strategy should be revised under these new circumstances. Identify which long-term goals should remain unchanged and which should be adjusted. Discuss the balance between emergency response and maintaining progress toward broader strategic objectives.

Switch perspectives and analyze the same situation from the viewpoints of five different stakeholders: a small business owner, a university student, a retired resident, the city's mayor, and an environmental scientist. For each stakeholder, explain their priorities, concerns, likely disagreements with others, and possible areas of compromise.

After completing the policy analysis, demonstrate creativity by writing a short narrative of approximately 300 words describing one ordinary day in Meridian ten years after your proposed plan has been implemented successfully. Focus on small details of everyday life rather than dramatic events. Allow readers to infer the city's transformation through subtle observations rather than direct exposition.

Next, perform a critical self-review of everything you have written. Identify at least five assumptions that influenced your recommendations. Discuss how those assumptions could be incorrect and how different assumptions might lead to different conclusions. Highlight any areas where additional evidence would improve confidence in your recommendations.

Then create a concise executive summary of no more than 250 words that could be presented to busy policymakers. The summary should capture the central findings, priorities, risks, and expected benefits without oversimplifying complex issues.

Following the executive summary, produce a comparison table with the following columns: Challenge, Root Cause, Recommended Action, Estimated Cost, Expected Benefit, Primary Risk, and Success Metric. Populate the table with information that is internally consistent with your earlier analysis.

Now demonstrate teaching ability by explaining the entire strategy twice. First, explain it as though speaking to a curious twelve-year-old using simple language, relatable analogies, and minimal jargon. Second, explain it as though presenting to graduate students specializing in urban planning, using precise terminology and greater technical detail while maintaining clarity.

Afterward, imagine that a skeptical journalist interviews you. Generate ten challenging questions that the journalist might ask about weaknesses, unintended consequences, political feasibility, fairness, environmental impacts, and financial realism. Answer each question thoughtfully, acknowledging uncertainty where appropriate instead of pretending perfect confidence.

Next, identify three alternative futures for Meridian. One should be highly optimistic, one moderately successful, and one pessimistic. Describe the events that lead to each outcome, the warning signs that policymakers should monitor, and the actions that could shift the city from a negative trajectory toward a positive one.

Demonstrate structured reasoning by listing the key principles that guided your recommendations. These principles might include resilience, equity, transparency, efficiency, adaptability, environmental stewardship, evidence-based policymaking, fiscal responsibility, or other relevant concepts. Explain how each principle influenced specific decisions within your strategy.

Throughout your response, prioritize logical consistency. If you introduce a claim, ensure that later recommendations do not contradict it without explanation. If uncertainty exists, state it explicitly. Distinguish clearly between facts provided in this prompt, reasonable inferences, assumptions, and speculative possibilities.

Your writing should balance analytical precision with readability. Use headings, numbered sections, bullet points where appropriate, and concise paragraphs. Avoid unnecessary repetition. When presenting estimates or projections, explain the reasoning instead of inventing unsupported numbers. Whenever multiple valid perspectives exist, compare them fairly before reaching a conclusion.

Finally, conclude with a reflective section discussing what this exercise reveals about the complexity of long-term decision-making in modern cities. Explain why simple solutions rarely solve interconnected problems and describe how effective leaders can make sound decisions despite uncertainty, incomplete information, competing priorities, and changing circumstances. End with three practical lessons that readers could apply to real-world planning, leadership, or problem-solving beyond the fictional city of Meridian.
""" ,
"""What do you know about the weather in antarctica"""

]

generation_event = threading.Event()
gen_time = 0

model_name = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def fun1(prompt, monitor_thread):


  global gen_time
  global start_time

  class TrackStreamer(BaseStreamer):
    def __init__(self, start_time):
      super().__init__()
      self.ttft = None
      self.sec_token_time = None
      self.start_time = start_time
      self.is_prompt = True       # <---- Unused
      self.is_first_token = True   # Clear flag for token 1
      self.is_second_token = True
      self.decode_start_time = None
      self.token_count = 0
      self.all_tokens = []

    def put(self, value):
      if value.ndim > 1 and value.shape[-1] > 1:
        torch.cuda.synchronize()
        self.start_time = time.time()

      elif self.is_first_token:
        torch.cuda.synchronize()
        self.ttft = time.time() - self.start_time
        self.decode_start_time = time.time()
        self.is_first_token = False
        self.all_tokens.append(value)

      elif self.is_second_token:
        torch.cuda.synchronize()
        self.sec_token_time = time.time() - self.decode_start_time
        # second_t = self.sec_token_time
        self.is_second_token = False
        self.token_count += value.numel()
        self.all_tokens.append(value)

      else:
        self.token_count += value.numel()
        self.all_tokens.append(value)

    def end(self):
      torch.cuda.synchronize()
      if self.decode_start_time and self.token_count > 0:
        decode_duration = time.time() - self.decode_start_time
        tokens_per_second = self.token_count / decode_duration
        print("TTFT -->", self.ttft)
        print("second_t -->", self.sec_token_time)
        print(f"Decode Speed: {tokens_per_second:.2f} tokens/sec")
        if self.all_tokens:
          flattened_tokens = torch.cat(self.all_tokens, dim=0)
          actual_response = tokenizer.decode(flattened_tokens, skip_special_tokens=True)

          print("\n=== Model Response ===")
          print(actual_response)
          print("======================\n")


  messages = [
  {"role": "system", "content": "You are a helpful AI assistant who provides correct information, always"},
  {"role": "user", "content": prompt}
  ]

  text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )

  model_inputs = tokenizer([text], return_tensors="pt").to(model.device)       #generate token IDs
  torch.cuda.synchronize()
  start_time = time.time()
  streamer = TrackStreamer(start_time=start_time)

  gen_time = start_time

  generation_event.set()
  monitor_thread.start()

  model.generate(
      **model_inputs,              #generate embeddings from token --> output response token??
      max_new_tokens=100,
      streamer=streamer,
  )
  generation_event.clear()
  torch.cuda.synchronize()
  end_time = time.time()
  print("Total exec time >>>", end_time - start_time )


def fun2():
  global arr1
  nvmlInit()
  device_count = nvmlDeviceGetCount()

  while generation_event.is_set():
    for i in range(device_count):
      arr_time = time.time()
      handle = nvmlDeviceGetHandleByIndex(i)
      name = nvmlDeviceGetName(handle)
      # driver_ver = nvmlSystemGetDriverVersion()

      memory_info = nvmlDeviceGetMemoryInfo(handle)
      #total_mem = memory_info.total / (1024**3)  # Convert to GB
      used_mem = memory_info.used / (1024**3)

      utilization = nvmlDeviceGetUtilizationRates(handle)
      gpu_util = utilization.gpu

      temp = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
      power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
      arr1.append([arr_time, used_mem, gpu_util, temp, power])
      time.sleep(0.1)


  nvmlShutdown()

def warmup(model, tokenizer, n_tokens=20):
  print("Warming up model...")
  messages = [{"role": "user", "content": "Hello"}]
  text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = tokenizer([text], return_tensors="pt").to(model.device)

  with torch.no_grad():
      model.generate(**inputs, max_new_tokens=n_tokens)

  torch.cuda.synchronize()
  print("Warm-up done.\n")



def main():
  global arr1
  all_results = []

  warmup(model, tokenizer)

  for prompt in prompts:
    arr1 = []

    t2 = threading.Thread(target=fun2)
    t1 = threading.Thread(target=fun1, args=(prompt, t2))

    t1.start()
    t1.join()
    t2.join()

    if len(arr1) > 0:
      arr = [row for row in arr1 if row[0] > gen_time]
      if len(arr) > 0:
          max_mem = max(row[1] for row in arr)
          max_gpu = max(row[2] for row in arr)
          avg_power = sum(row[4] for row in arr) / len(arr)

          # Store the metrics for this prompt
          all_results.append({
              "prompt": prompt,
              "max_mem": max_mem,
              "max_gpu": max_gpu,
              "avg_power": avg_power
          })
      else:
        print(f"No GPU samples after gen_time for prompt: {prompt[:30]}")
    else:
      print("Data not found!!!")

  # Print comparison report
  print("\n==================== BENCHMARK COMPARISON REPORT ====================")
  print(f"{'Prompt Summary':<30} | {'Max Mem (GB)':<12} | {'Max GPU (%)':<11} | {'Avg Power (W)':<13}")
  print("-" * 75)

  # Truncate prompt string if it's too long for the table column

  for res in all_results:
    short_prompt = res['prompt'] if len(res['prompt']) < 28 else res['prompt'][:25] + "..."
    print(f"{short_prompt:<30} | {res['max_mem']:<12.2f} | {res['max_gpu']:<11.1f} | {res['avg_power']:<13.2f}")
  print("=====================================================================")


if __name__ == "__main__":
  main()
