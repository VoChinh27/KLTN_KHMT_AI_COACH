'use strict';
const $ = id => document.getElementById(id);
let csrf = '', uploads = [], jobs = [], current = '', busy = false, submitting = false, uploading = false;
let previewURL = null, displayedResult = null;
const labels = {metadata:'Đọc metadata và kiểm tra nguồn',pose:'Pose / cache',quality:'Kiểm tra chất lượng',preprocessing:'Tiền xử lý',geometry:'Đo hình học',save:'Lưu kết quả',render:'Tạo video overlay',video_validation:'Giải mã kiểm tra video',finalize:'Ghi manifest',pipeline:'Luồng phân tích'};
const states = {idle:'Sẵn sàng',running:'Đang phân tích',verifying:'Đang xác minh artifacts',completed:'Hoàn tất',failed:'Lỗi xử lý',finishing_failure:'Đang lưu lỗi'};
function fail(message) { $('error').hidden=false; $('error').textContent=message; }
async function api(path, options={}) {
  const response=await fetch(path,{...options,headers:{...(options.headers||{}),...(options.method?{'X-Gym-CSRF':csrf}:{})}});
  const data=await response.json(); if(!response.ok) throw new Error(data.message||data.error); return data;
}
function controls() {
  for(const id of ['file','upload','side','mode','cache']) $(id).disabled=busy||submitting||uploading;
  $('ack').disabled=busy||submitting||uploading||$('mode').value!=='exploratory';
  $('start').disabled=busy||submitting||uploading||!$('upload').value||($('mode').value==='exploratory'&&!$('ack').checked);
}
function option(select,value,label) { const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o); }
function metadata() {const u=uploads.find(x=>x.upload_id===$('upload').value);$('metadata').textContent=u?`${u.original_name} · ${u.metadata.width} × ${u.metadata.height} · ${u.metadata.fps} FPS · ${u.metadata.frame_count} frames · SHA-256 ${u.sha256}`:'';controls();}
function artifactURL(name,download=false) {return `/artifacts/${current}/${encodeURIComponent(name)}${download?'?download=1':''}`;}
function showJob(job) {
  if(!job) return;
  current=job.job_id;location.hash=current;
  $('jobs').value=current;
  const e=job.event;
  $('stage').textContent=job.status==='running'&&e?(labels[e.stage]||e.stage):(states[job.status]||job.status);
  $('elapsed').textContent=`${(job.active?job.elapsed_seconds:(e?.elapsed_seconds||0)).toFixed(1)} s`;
  if(job.active&&e&&e.total!==null) {$('progress').max=e.total;$('progress').value=e.completed;$('counts').textContent=`${e.completed} / ${e.total} ${e.unit} — riêng stage hiện hành`;} else if(job.active) {$('progress').removeAttribute('value');$('counts').textContent='Đang xử lý; stage hiện tại chưa có mẫu số tiến độ.';} else {$('progress').max=1;$('progress').value=0;$('counts').textContent='Không có phần trăm tổng pipeline.';}
  $('cacheState').textContent=`Pose cache: ${job.cache_status||'chưa kiểm tra'}`;
  $('log').textContent=job.log.join('\n');$('evidence').textContent=`Evidence: ${job.evidence_dir}`;
  $('message').textContent=`Job ${current} · gối ${job.side==='left'?'trái':'phải'} cố định · ${job.mode}`;
  if(job.error) fail(job.error.message);
  const ready=job.status==='completed'&&!job.active&&job.coverage!==null;
  $('result').hidden=!ready;
  if(ready) {
    const c=job.coverage;$('coverage').textContent=`${c.available_frames} / ${c.total_frames}`;
    $('resultNote').textContent=c.available_frames===0?'Xử lý hoàn tất — chưa đủ điều kiện đo. Các giá trị bị chặn giữ null và lý do.':`Độ phủ số đo ${(c.available_ratio*100).toFixed(1)}%; không phải accuracy. ${c.unavailable_frames} frame không đủ điều kiện đo.`;
    if(displayedResult!==current) {$('overlay').src=artifactURL(job.video_file);$('report').href=artifactURL('START_HERE.html');$('artifact').replaceChildren();for(const name of job.artifacts) option($('artifact'),name,name);$('artifact').value='result.json';$('download').href=artifactURL($('artifact').value,true);displayedResult=current;}
  }
}
async function refresh(initial=false) {
  try {
    const state=await api('/api/state');csrf=state.csrf_token;uploads=state.uploads;jobs=state.jobs;
    busy=jobs.some(j=>j.active); const chosen=$('upload').value;
    $('upload').replaceChildren();option($('upload'),'','Chọn video đã tiếp nhận');
    for(const u of uploads) option($('upload'),u.upload_id,u.original_name);$('upload').value=uploads.some(u=>u.upload_id===chosen)?chosen:(initial?(uploads.at(-1)?.upload_id||''):'');
    $('jobs').replaceChildren();option($('jobs'),'','Chưa chọn job');for(const j of jobs) option($('jobs'),j.job_id,`${j.job_id.slice(0,8)} · ${j.mode} · ${states[j.status]||j.status}`);
    if(initial) {const id=location.hash.slice(1);current=jobs.some(j=>j.job_id===id)?id:(jobs.find(j=>j.active)?.job_id||jobs.at(-1)?.job_id||'');if(uploads.length)$('uploadStatus').textContent='Đã khôi phục danh sách video tiếp nhận trong phiên server này.';}
    showJob(jobs.find(j=>j.job_id===current));metadata();controls();
  } catch(e) {fail('Không kết nối được server: '+e.message);$('start').disabled=true;}
}
$('mode').addEventListener('change',()=>{$('ack').checked=false;controls();});$('ack').addEventListener('change',controls);
$('upload').addEventListener('change',()=>{if(previewURL){URL.revokeObjectURL(previewURL);previewURL=null;}$('preview').removeAttribute('src');$('preview').hidden=true;$('file').value='';$('error').hidden=true;$('uploadStatus').textContent=$('upload').value?'Đang dùng video đã tiếp nhận và kiểm tra decoder.':'Chưa chọn video.';metadata();});
$('file').addEventListener('change',async()=>{
  const file=$('file').files[0];if(!file)return;$('error').hidden=true;
  $('upload').value='';metadata(); // failed upload must not silently select an older video
  if(file.size>250*1024*1024||file.size===0){fail('Video phải từ 1 byte đến 250 MiB.');return;}
  if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(file);$('preview').src=previewURL;$('preview').hidden=false;
  $('previewNote').textContent='Preview phụ thuộc codec trình duyệt; backend sẽ kiểm tra decoder riêng.';
  uploading=true;controls();$('uploadStatus').textContent='Đang tiếp nhận video và kiểm tra decoder…';
  try {const u=await api('/api/uploads',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-File-Name':encodeURIComponent(file.name)},body:file});await refresh();$('upload').value=u.upload_id;metadata();$('uploadStatus').textContent='Video đã được tiếp nhận. Decoder đọc được metadata và frame đầu; toàn timeline kiểm tra trong pipeline.';}
  catch(e){$('uploadStatus').textContent='Video chưa được chấp nhận.';fail(e.message);}
  finally{uploading=false;controls();}
});
$('preview').addEventListener('error',()=>{$('previewNote').textContent='Trình duyệt không preview được codec này. Xem trạng thái decoder; đây chưa phải lỗi pipeline.';});
$('start').addEventListener('click',async()=>{
  if(busy||submitting||uploading)return;submitting=true;controls();$('error').hidden=true;
  try {const job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({upload_id:$('upload').value,request_id:crypto.randomUUID().replaceAll('-',''),side:$('side').value,mode:$('mode').value,acknowledged:$('ack').checked,use_cache:$('cache').checked})});current=job.job_id;await refresh();}
  catch(e){fail(e.message);}finally{submitting=false;controls();}
});
$('jobs').addEventListener('change',()=>{current=$('jobs').value;$('error').hidden=true;showJob(jobs.find(j=>j.job_id===current));});
$('artifact').addEventListener('change',()=>{$('download').href=artifactURL($('artifact').value,true);});
(async()=>{await refresh(true);async function poll(){await refresh();setTimeout(poll,700);}setTimeout(poll,700);})();
