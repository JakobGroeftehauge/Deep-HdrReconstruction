import ffmpeg
import logging

from dataclasses import dataclass


def create_logger(name): 
    logging.basicConfig(filename=name, format='%(asctime)s:%(msecs)03d: %(levelname)s: %(name)s: %(message)s',level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    return logger

def setup_decoder(input_pth, width, height): 
    decoder = (ffmpeg.input(input_pth).output('pipe:', format='rawvideo', loglevel='quiet', pix_fmt='rgb24', s='{}x{}'.format(width, height)).run_async(pipe_stdout=True))
    return decoder

def setup_encoder(output_pth, width, height, fps, MAX_LUM=1000): 
    encoder = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb48le', s='{}x{}'.format(width, height), framerate=fps, color_primaries='bt709', color_trc='smpte2084', colorspace='bt709') 
        .output(output_pth, pix_fmt='yuv420p10le', loglevel='quiet', **{'crf':0, 'c:v': 'libx265', 'x265-params': 'keyint=25:bframes=2:vbv-bufsize=50000:vbv-maxrate=50000:hdr-opt=1:no-open-gop=1:hrd=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,100):max-cll={},300'.format(MAX_LUM)})
        .overwrite_output()
        .run_async(pipe_stdin=True))
    return encoder

def get_output_path(string, half=False): 
    path_list = string.split("/")
    if half: 
        file_ending = "_DeepHDR.mkv"
    else: 
        file_ending = "_DeepHDR_fp16.mkv"
    path_list[-1] = path_list[-1].split(".")[0] + "_DeepHDR.mkv"
    return "/".join(path_list)


@dataclass
class PipelineParams:
    model_pth: str
    input_pth: str

    output_pth: str = None
    width: int = None
    height: int = None
    N_numbers: int = 6  # Number of 
    disable_model: bool = False      # disable onnx calls, used for debugging on hardware without GPU. 
    half: bool = False      # disable onnx calls, used for debugging on hardware without GPU. 
    logger_name: str = "debug.log"  # Name of the debug log
    sc: float = 20 #
    max_luminance: int = 1000

    # Automatically initalised variables:
    fps: float = None
    n_frames: int = None # number of frames in the video
    size: int = None # Number of data elements in a frame
    arr_shape: list = None # Shape of a frame

    def __post_init__(self):
        w, h, fps, self.n_frames = self.extract_video_data()
        if self.width == None or self.height == None:
            self.width = w
            self.height = h
        if self.fps == None: 
            self.fps = fps
        
        self.size = self.width * self.height * 3
        self.arr_shape = [3, self.height, self.width]
        if self.output_pth == None: 
            self.output_pth = get_output_path(self.input_pth, half=self.half)

    def extract_video_data(self): 
      probe = ffmpeg.probe(self.input_pth)
      video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
      width = int(video_stream['width'])
      height = int(video_stream['height'])
      fps = int(video_stream['r_frame_rate'].split('/')[0])
      num_frames= int(video_stream['nb_frames'])
      return width, height, fps, num_frames
    
    
