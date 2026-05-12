import os
import cv2
import h5py
import requests
import os
import socket
from urllib.parse import urlparse
import time
from datetime import datetime
from textual.app import App, ComposeResult
from textual.widgets import Input, Button, Label, Static, Select
from textual.containers import Vertical
from IPython.display import display, clear_output

cube_description = [
    "glass", "brightly colored plastic", "wood", "steel", "worn dull colored", "cheese",
    "polished brass", "matte black", "translucent acrylic", "rustic copper", "brushed aluminum",
    "painted ceramic", "textured rubber", "carbon fiber", "iridescent metal", "frosted crystal",
    "ice", "origami paper"
]
tray_description = [
    "glass", "brightly colored plastic", "wood", "steel", "worn dull colored", "cheese",
    "polished brass", "matte black", "translucent acrylic", "rustic copper", "brushed aluminum",
    "painted ceramic", "textured rubber", "carbon fiber", "iridescent metal", "frosted crystal",
    "ice", "origami paper"
]
table_material = [
    "wood", "marble", "metal", "glass", "stone",
    "polished stainless steel", "concrete", "brushed aluminum", "tempered glass", 
    "granite", "composite resin", "industrial plastic", "carbon fiber"
]
location = [
    "cluttered workshop", "manufacturing facility", "university laboratory", "restaurant",
    "high-tech cleanroom", "industrial warehouse", "robotics research center", 
    "automated assembly line", "testing facility", "engineering lab", 
    "technology demonstration room", "quality control station"
]

class VideoForm(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #form {
        width: 60%;
        padding: 2;
        border: solid green;
    }
    Input {
        margin-bottom: 1;
    }
    Button {
        width: 20;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="form"):
            yield Label("Enter URL:")
            yield Input(placeholder="http://192.168.207.23:5005", id="url")

            # yield Label("Dataset Path:")
            # yield Input(placeholder="/path/to/dataset", id="dataset_path")

            # yield Label("Output Path:")
            # yield Input(placeholder="/path/to/output", id="output_path")
            
            yield Label("Seed:")
            yield Input(placeholder="42", id="seed")
            
            yield Label("Control weight:")
            yield Input(placeholder="0.4", id="control_weight")
            
            yield Label("Sigma Max:")
            yield Input(placeholder="40", id="sigma_max")
            
            yield Label("Canny Strength:")
            yield Input(placeholder="Very Low", id="cannny_strength")
            
            yield Label("Cube description:")
            yield Select(
                options=[(option, option) for option in cube_description]  ,
                id="cube_select"
            )
            
            yield Label("Tray description:")
            yield Select(
                options=[(option, option) for option in tray_description]  ,
                id="tray_select"
            )
            
            yield Label("Table material:")
            yield Select(
                options=[(option, option) for option in table_material]  ,
                id="table_select"
            )
            
            yield Label("Locaton:")
            yield Select(
                options=[(option, option) for option in location]  ,
                id="location_select"
            )          
            
            yield Button("Submit", id="submit")

            yield Static("", id="output")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            url = self.query_one("#url", Input).value
            # dataset_path = self.query_one("#dataset_path", Input).value
            # output_path = self.query_one("#output_path", Input).value
            seed_str = self.query_one("#seed", Input).value or ""
            control_weight_str = self.query_one("#control_weight", Input).value or ""
            sigma_max_str = self.query_one("#sigma_max", Input).value or ""
            canny_strength = (self.query_one("#cannny_strength", Input).value or "medium").strip()

            try:
                seed = int(seed_str)
            except Exception:
                seed = 42

            try:
                control_weight = float(control_weight_str)
            except Exception:
                control_weight = 0.5

            try:
                sigma_max = float(sigma_max_str)
            except Exception:
                sigma_max = 70.0

            # selections
            cube_desc = self.query_one("#cube_select", Select).value
            tray_desc = self.query_one("#tray_select", Select).value
            table_mat = self.query_one("#table_select", Select).value
            location_desc = self.query_one("#location_select", Select).value

            # assemble a simple prompt from the selected descriptors
            prompt =    f"The scene depicts a robotic arm performing a precise cubes-collecting operation with {cube_desc} \
                        cubes to a {tray_desc} tray in a {location_desc}. The workspace is centered on a {table_mat} table. On the table are cubes. The robotic \
                        arm, featuring a sleek industrial design clad in smooth black plastic with metallic joints, \
                        extends from its base to manipulate the cubes. The entire scene is illuminated by professional studio lighting, \
                        creating soft shadows. A clutter of equipment can be seen in the background. The camera is fixed in place."

            # # Display the collected input
            msg = (
                f"\nURL: {url}\n"
                f"\seed: {seed}\n"
                f"\control_weight: {control_weight}\n"
                f"\sigma_max: {sigma_max}\n"
                
                # f"Dataset Path: {dataset_path}\n"
                # f"Output Path: {output_path}"
            )
            self.query_one("#output", Static).update(msg)
            
            params = {
                # "dataset_path": dataset_path,
                # "output_path": output_path,
                "url": url,
                "seed_str": seed_str,
                "control_weight_str": control_weight_str,
                "sigma_max_str": sigma_max_str,
                "canny_strength": canny_strength,
                "prompt": prompt,         
                }
            
            self.exit(params) 



def video_generator_from_hdf5(dataset_dir: str, output_path:str, fps: int, episode: int, camera_name: str ) -> None:
    """Encode a sequence of frames into a video.
    """
    dataset = os.path.join(dataset_dir, f"episode_{episode}.hdf5")
    print(dataset)
    with h5py.File(dataset, 'r') as root:
        image_frames  = root[f'/images/colors/{camera_name}'][()]  
        # image_1_frames  = root['/images/colors/cam1'][()]  
        # image_2_frames  = root['/images/colors/cam2'][()] 
        # depth_1_frames  = root['/images/depths/cam1'][()]   
        # depth_2_frames  = root['/images/depths/cam2'][()] 
        
        
    frames, H, W, _ = image_frames.shape
    # assert image_1_frames.shape == image_2_frames.shape, f"images from each camera does not match: cam1={image_1_frames.shape} ,cam2={image_1_frames.shape}"
    
    # --- RGB video writer ---
    rgb_writer = cv2.VideoWriter(os.path.join(output_path,f"episode_{episode}_{camera_name}_rgb.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )
    
    '''# --- Depth video writer ---
    depth_writer = cv2.VideoWriter(
        "episode_1_cam1_depth_control.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
        isColor=False,
    )'''
    for t in range(frames):
        rgb_writer.write(cv2.cvtColor(image_frames[t], cv2.COLOR_RGB2BGR))
        #depth_gray = depth_norm[t]
        #depth_writer.write(depth_gray)
    
    rgb_writer.release()
    # depth_writer.release()       
    

def test_connection(url):
    """Test basic connectivity to the server."""
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
    
    try:
        # Try to establish a TCP connection
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        return True, None
    except socket.timeout:
        return False, f"Connection timed out when trying to reach {host}:{port}"
    except socket.gaierror:
        return False, f"DNS resolution failed for {host}"
    except ConnectionRefusedError:
        return False, f"Connection refused by {host}:{port}"
    except Exception as e:
        return False, f"Connection test failed: {str(e)}"

def process_video(
    video_path: str,
    output_path: str,
    url: str,
    prompt: str,
    sigma_max: float = 70,
    control_weight: float = 0.5,
    canny_strength: str = "medium",
    seed: int = 42,
    poll_interval: int = 10,
    max_poll_time: int = 3600,
) -> requests.Response:
    """
    Process a video using the NVCF API.

    Args:
        url (str): The base URL of the NVCF API.
        video_path (str): The path to the video file to process.
        output_path (str): The path to save the processed video.
        prompt (str): The prompt to use to condition the Cosmos model.
        sigma_max (float): The maximum sigma value.
        control_weight (float): Controls how strongly the control input should affect the output.
        canny_strength (str): The strength of the canny edge detection.
        seed (int): The seed for the random number generator.
        poll_interval (int): How often to poll for job completion (seconds).
        max_poll_time (int): Maximum time to wait for job completion (seconds).
    """

    if not os.path.exists(video_path):
        raise ValueError(f"rocess_video(): Video file not found at {video_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Test connection first
    print(f"Testing connection to {url}...")
    success, message = test_connection(url)
    if not success:
        print(f"Connection test failed: {message}")
        print("Please check:")
        print("1. If the server is running")
        print("2. If the IP address and port are correct")
        print("3. If there are any firewalls blocking the connection")
        print("4. If the server is accessible from your network")
        return None

    # Parameters for the request
    params = {
        "prompt": f"\"{prompt}\"",
        "sigma_max": sigma_max,
        "control_weight": control_weight,
        "canny_strength": canny_strength.lower().replace(" ", "_"),
        "seed": seed,
    }
    
    submit_url = f"{url}/canny/submit"
    status_url = f"{url}/canny/status"
    result_url = f"{url}/canny/result"
    
    try:
        # Submit the job
        print("Submitting video processing job...")
        files = {
            'video': ('video.mp4', open(video_path, 'rb'), 'video/mp4')
        }
        
        try:
            # Submit the job with a shorter timeout
            response = requests.post(
                submit_url,
                data=params,
                files=files,
                timeout=90,
                verify=False
            )
            
            if response.status_code != 200:
                print(f"Error submitting job: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error: {error_data.get('error', 'Unknown error')}")
                except ValueError:
                    print(f"Error details: {response.text}")
                return None
            
            # Get job ID from response
            job_data = response.json()
            job_id = job_data['job_id']
            print(f"Job submitted successfully. Job ID: {job_id}")
            
            # Poll for completion
            start_time = time.time()
            while True:
                if time.time() - start_time > max_poll_time:
                    print(f"Maximum polling time ({max_poll_time}s) exceeded")
                    return None
                
                try:
                    status_response = requests.get(
                        f"{status_url}/{job_id}",
                        timeout=30,
                        verify=False
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data['status']
                        
                        if status == 'completed':
                            print("Processing completed successfully!")
                            break
                        elif status == 'failed':
                            print(f"Processing failed: {status_data.get('error', 'Unknown error')}")
                            return None
                        else:
                            print(f"{datetime.fromtimestamp(time.time())} | Status: {status}")
                    else:
                        print(f"Error checking status: HTTP {status_response.status_code}")
                        return None
                        
                except requests.RequestException as e:
                    print(f"Error checking status: {e}")
                    # Continue polling despite temporary errors
                
                time.sleep(poll_interval)
            
            # Download the result
            print("Downloading processed video...")
            result_response = requests.get(
                f"{result_url}/{job_id}",
                timeout=90,
                verify=False,
                stream=True
            )
            
            if result_response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in result_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Video successfully downloaded and saved to {output_path}")
                return result_response
            else:
                print(f"Error downloading result: HTTP {result_response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timed out")
        except requests.exceptions.SSLError:
            print("SSL/TLS error occurred. If using self-signed certificates, verify=False is already set.")
        except requests.exceptions.ConnectionError as e:
            print(f"Failed to connect to the server: {e}")
        except requests.RequestException as e:
            print(f"Connection error: {e}")
            
    finally:
        # Ensure the file is closed
        if 'files' in locals():
            files['video'][1].close()
    
    return None 



if __name__ == '__main__':
    app = VideoForm()
    params = app.run()
    if params is None:
        print("Form was cancelled or closed without submitting.")
    else:
        # dataset_path    =params["dataset_path"]
        url             =params["url"]
        # output_path     =params["output_path"]
        seed            =params["seed_str"]
        control_weight  =params["control_weight_str"]
        sigma_max       =params["sigma_max_str"]
        canny_strength  =params["canny_strength"]
        prompt          =params["prompt"]

    DATASET_DIR     = "/mnt/bigdata/00_students/wee_ucl/seed_28"  #dataset_path
    output_path     = "/mnt/bigdata/00_students/wee_ucl/VRWIT/output"
    
    episodes = [d for d in os.listdir(DATASET_DIR)] 
    print(f"size of dataset: {len(episodes)}")

    for ep in range(2):  #len(episodes)-1
        for cam in ["cam1","cam2"]:
            video_generator_from_hdf5(dataset_dir=DATASET_DIR, output_path=output_path, fps=24, episode=ep, camera_name=cam)
            video_path=os.path.join(output_path,f"episode_{ep}_{cam}_rgb.mp4")
            output_path=os.path.join(output_path,f"generated_episode_{ep}_{cam}_rgb.mp4")
            response = process_video(video_path,output_path,url, prompt, sigma_max, control_weight, canny_strength, seed)     #, sigma_max, control_weight, canny_strength, seed)
            # response = process_video(video_path=video_path, **params)
            if response is None:
                display("An error occurred processing the request")
            elif response.status_code == 200:
                clear_output(wait=True)
                # display(Video(output_path))
                # display(create_download_link(output_path, link_text=f"Download Video: {output_path}"))
        