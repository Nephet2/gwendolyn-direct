"""Read-only Restim sensor perception shared by Gwendolyn Vector and Signal Lab."""
from __future__ import annotations
import base64, hashlib, json, math, os, queue, socket, struct, threading, time
from collections import deque
from statistics import median, pstdev


class Reader(threading.Thread):
    def __init__(self, name, path, output):
        super().__init__(daemon=True, name=f"Restim{name}Reader")
        self.name_, self.path, self.output = name, path, output
        self.stop_event, self.sock = threading.Event(), None
    def exact(self, n):
        data=bytearray()
        while len(data)<n:
            part=self.sock.recv(n-len(data))
            if not part: raise ConnectionError("closed")
            data.extend(part)
        return bytes(data)
    def pong(self, payload):
        mask=os.urandom(4); body=bytes(v^mask[i%4] for i,v in enumerate(payload))
        self.sock.sendall(bytes([0x8A,0x80|len(payload)])+mask+body)
    def connect(self):
        s=socket.create_connection(("localhost",12346),timeout=3); s.settimeout(2)
        key=base64.b64encode(os.urandom(16)).decode()
        req=(f"GET {self.path} HTTP/1.1\r\nHost: localhost:12346\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
             f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        s.sendall(req.encode()); response=bytearray()
        while b"\r\n\r\n" not in response: response.extend(s.recv(4096))
        head=bytes(response).split(b"\r\n\r\n",1)[0].decode("latin1")
        expected=base64.b64encode(hashlib.sha1((key+"258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        if " 101 " not in head.splitlines()[0] or expected.lower() not in head.lower(): raise ConnectionError("handshake rejected")
        self.sock=s
    def run(self):
        # Restim may start after Gwendolyn, and Windows can briefly interrupt a
        # local WebSocket when a device reconnects. Keep recovering until the
        # application explicitly closes the reader.
        while not self.stop_event.is_set():
            try:
                self.connect(); self.output.put(("status",self.name_,True,None))
                while not self.stop_event.is_set():
                    try: a,b=self.exact(2)
                    except socket.timeout: continue
                    op=a&15; length=b&127
                    if length==126:length=struct.unpack("!H",self.exact(2))[0]
                    elif length==127:length=struct.unpack("!Q",self.exact(8))[0]
                    mask=self.exact(4) if b&128 else None; data=self.exact(length)
                    if mask:data=bytes(v^mask[i%4] for i,v in enumerate(data))
                    if op==8:break
                    if op==9:self.pong(data[:125]);continue
                    if op not in (1,2):continue
                    try: value=json.loads(data.decode("utf-8"))
                    except Exception: continue
                    self.output.put(("data",self.name_,time.time(),value))
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.output.put(("status",self.name_,False,str(exc)))
            finally:
                self._close_socket()
            if not self.stop_event.is_set():
                self.stop_event.wait(2.0)

    def _close_socket(self):
        if self.sock:
            try:self.sock.close()
            except Exception:pass
            self.sock=None

    def close(self):
        self.stop_event.set()
        self._close_socket()


class RestimSensorBridge:
    """Converts raw streams into session-relative observations; never writes to Restim."""
    def __init__(self):
        self.q=queue.Queue(); self.lock=threading.Lock(); self.stop_event=threading.Event()
        self.samples={"IMU":deque(maxlen=1600),"AS5311":deque(maxlen=1400)}
        self.connected={"IMU":False,"AS5311":False}; self.errors={}
        self.started=time.time(); self.last_observation=""; self.observation_since=0.0
        self.readers=[Reader("IMU","/sensors/imu",self.q),Reader("AS5311","/sensors/as5311",self.q)]
        for r in self.readers:r.start()
        threading.Thread(target=self._pump,daemon=True,name="RestimSensorAggregator").start()
    def _pump(self):
        while not self.stop_event.is_set():
            try:event=self.q.get(timeout=.5)
            except queue.Empty:continue
            with self.lock:
                if event[0]=="status":
                    self.connected[event[1]]=event[2]
                    if event[3]:self.errors[event[1]]=event[3]
                    elif event[2]:self.errors.pop(event[1],None)
                else:self.samples[event[1]].append((event[2],event[3]))
    def close(self):
        self.stop_event.set()
        for r in self.readers:r.close()
    def snapshot(self):
        now=time.time()
        with self.lock:
            imu=list(self.samples["IMU"]); cal=list(self.samples["AS5311"]); connected=dict(self.connected); errors=dict(self.errors)
        recent_i=[v for t,v in imu if now-t<=2 and isinstance(v,dict)]
        recent_c=[v for t,v in cal if now-t<=2 and isinstance(v,dict) and isinstance(v.get("x"),(int,float))]
        fresh_i=bool(imu and now-imu[-1][0]<=2); fresh_c=bool(cal and now-cal[-1][0]<=2)
        gyro=[math.sqrt(sum(float(v.get(k,0))**2 for k in ("gyr_x","gyr_y","gyr_z"))) for v in recent_i]
        gyro_rms=math.sqrt(sum(x*x for x in gyro)/len(gyro)) if gyro else 0
        movement="unavailable" if not fresh_i else "still" if gyro_rms<.04 else "gentle" if gyro_rms<.16 else "active" if gyro_rms<.32 else "vigorous"
        xs=[float(v["x"])*1000 for v in recent_c]
        cal_range=max(xs)-min(xs) if xs else 0; cal_sd=pstdev(xs) if len(xs)>1 else 0
        clench="unavailable" if not fresh_c else "quiet" if cal_range<.05 else "subtle" if cal_range<.18 else "distinct" if cal_range<.40 else "strong"
        # Baseline is descriptive only: median of the first three seconds available.
        base=[float(v.get("x"))*1000 for t,v in cal if t-self.started<=3 and isinstance(v,dict) and isinstance(v.get("x"),(int,float))]
        relative=(median(xs)-median(base)) if xs and base else None
        position_state=("unavailable" if relative is None else
                        "increased" if relative >= .10 else
                        "decreased" if relative <= -.10 else "near session baseline")
        response_cues=[]
        if movement not in {"unavailable","still"}:
            response_cues.append("hip thrust / arousal response")
        if clench not in {"unavailable","quiet"}:
            response_cues.append("induced clenching")
        if position_state=="increased":
            response_cues.append("growth / sustained clenching")
        assumed_meaning="positive response" if response_cues else "quiet baseline"
        observation=f"movement {movement}; clenching {clench}"
        if observation!=self.last_observation:
            self.last_observation, self.observation_since=observation, now
        return {"read_only":True,"fresh":fresh_i or fresh_c,"connections":connected,"errors":errors,
                "movement":{"state":movement,"gyro_rms":round(gyro_rms,4),"fresh":fresh_i},
                "clenching":{"state":clench,"two_second_range_mm":round(cal_range,3),"variation_mm":round(cal_sd,3),
                              "relative_to_session_start_mm":round(relative,3) if relative is not None else None,
                              "position_state":position_state,"fresh":fresh_c},
                "response_interpretation":{"default":assumed_meaning,"cues":response_cues,
                    "convention":"session-relative; user correction overrides"},
                "observation":observation,"stable_seconds":round(max(0,now-self.observation_since),1)}
