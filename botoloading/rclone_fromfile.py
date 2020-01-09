#!/usr/local/anaconda3/bin/python
"""
We are going to keep reloading the file and download while more files are appended.
"""
import time
def init(rclone_config_fname):
    rclone_config_file = """
        [publicAWS]
        type = s3
        provider = AWS
        env_auth =
        access_key_id =
        secret_access_key =
        region =
        endpoint =
        location_constraint =
        acl =
        server_side_encryption =
        storage_class =
        """
    # can't stand not indenting so this clears the indentation
    rclone_config_file = '\n'.join([line.strip() for line in rclone_config_file.split('\n')])
    with open(rclone_config_fname, 'w') as fid:
        fid.write(rclone_config_file)


def follow(filepointer):
    #filepointer.seek(0,2)
    while True:
        line = filepointer.readline()
        if not line:
            time.sleep(0.5)
            continue
        yield line    


def main():
    rclone_config_fname = 'rclone.config'
    init(rclone_config_fname)
    filename = 'rclone_granules.txt'
    num_lines = 0
    with open(filename, 'r') as fid:
        for line in follow(fid):
            cmd = 'rclone copy --config=' + rclone_config_fname + ' copy ' + line + ' ./data'
            print(cmd)

    print('Done')

if __name__ == "__main__":
    main()