import os

def create_video_folder(video_path, parent_dir='Wrong Frame'):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_subdir = os.path.join(parent_dir, video_name)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    return output_subdir