"""Prepare source-image sheets only; extraction never claims visual review."""
import argparse
import json
from pathlib import Path

from gym_coach.counting.readiness import file_info, write_json
from gym_coach.video.reader import OpenCVVideoReader


def extract(source, output, *, step=1, width=160, page_size=100, columns=10):
    import cv2
    import numpy as np
    path=Path(source['path'])
    if (not path.resolve().is_relative_to(Path.cwd().resolve()/'Data/UCF-101_GYM/BodyWeightSquats')
            or file_info(path)['sha256']!=source['sha256']):
        raise ValueError('Source identity mismatch')
    if any(type(v) is not int or v<1 for v in (step,width,page_size,columns)):
        raise ValueError('Positive integer sheet parameters required')
    reader=OpenCVVideoReader(path)
    metadata=reader.metadata
    if any(getattr(metadata,k)!=source[k] for k in ('video_id','fps','frame_count','width','height')):
        raise ValueError('Decoder timeline/size differs from frozen source')
    output=Path(output)
    settings=dict(step=step,width=width,page_size=page_size,columns=columns)
    if (output/'manifest.json').is_file():
        previous=json.loads((output/'manifest.json').read_bytes())
        if previous['source']!=source or previous['settings']!=settings:
            raise ValueError('Existing sheet identity mismatch')
        for sheet in previous['sheets']:
            if file_info(output/sheet['file'])!=sheet['identity']:
                raise ValueError('Existing sheet bytes changed')
        return previous
    output.mkdir(parents=True,exist_ok=False)
    height=round(width*source['height']/source['width']); cell_height=height+22
    sheets=[];tiles=[];indices=[];decoded=0
    def flush():
        canvas=np.full((40+((len(tiles)+columns-1)//columns)*cell_height,columns*width,3),255,dtype=np.uint8)
        cv2.putText(canvas,f"{source['video_id']} | SOURCE FRAMES | sheet {len(sheets)+1}",(8,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
        for j,(tile,index) in enumerate(zip(tiles,indices)):
            x=(j%columns)*width;y=40+(j//columns)*cell_height
            canvas[y:y+height,x:x+width]=tile
            cv2.putText(canvas,f"f{index:03}  {index/source['fps']:.2f}s",(x+3,y+height+16),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
        name=f'sheet-{len(sheets):03}.jpg'
        if not cv2.imwrite(str(output/name),canvas,[cv2.IMWRITE_JPEG_QUALITY,94]):raise OSError('Sheet write failed')
        sheets.append(dict(file=name,identity=file_info(output/name),frame_indices=list(indices),
                           timestamps_seconds=[i/source['fps'] for i in indices]))
        tiles.clear();indices.clear()
    for frame in reader.frames():
        if frame.frame_index!=decoded or frame.timestamp_seconds!=decoded/source['fps']:
            raise ValueError('Decoder frame/time mismatch')
        decoded+=1
        if frame.frame_index%step==0 or frame.frame_index==source['frame_count']-1:
            tiles.append(cv2.resize(frame.image,(width,height),interpolation=cv2.INTER_AREA));indices.append(frame.frame_index)
            if len(tiles)==page_size:flush()
    if tiles:flush()
    if decoded!=source['frame_count']:raise ValueError('Incomplete source decode')
    result=dict(version='squat-ai-source-sheets-v1',source=source,settings=settings,
        decoded_frames=decoded,prepared_not_observed=True,transform='Uniform resize, JPEG contact-sheet layout only; no crop/flip/rotation/pose overlay',
        sheets=sheets,decoder=metadata.decoder,timestamp_policy=source['timestamp_policy'])
    write_json(output/'manifest.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--video',required=True)
    p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--step',type=int,default=1)
    a=p.parse_args();dataset=json.loads(a.manifest.read_bytes())
    source=next(r['source'] for r in dataset['videos'] if r['source']['video_id']==a.video)
    if not a.output_dir.resolve().is_relative_to(Path.cwd().resolve()/'outputs'):raise ValueError('New output under outputs required')
    result=extract(source,a.output_dir,step=a.step)
    print(json.dumps(dict(video=a.video,frames=result['decoded_frames'],sheets=len(result['sheets']),visually_reviewed=False)))


if __name__=='__main__':main()
