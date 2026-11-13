"""Launch R continuation once, only after generation and full audit succeed."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    source=Path(sys.argv[1]).resolve()
    status_path=source/'phase2_autostart.json'
    lock=source/'phase2_autostart.lock'
    with lock.open('x') as f:
        f.write(datetime.datetime.now().isoformat())
    status={'state':'waiting_for_generation','source':str(source)}

    def save():
        status['updated']=datetime.datetime.now().isoformat()
        temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status,indent=2),encoding='utf-8')
        temp.replace(status_path)

    save()
    try:
        deadline=time.monotonic()+6*3600
        while time.monotonic()<deadline:
            generation=json.loads((source/'generation_status.json').read_text())
            if generation['state']=='failed':
                raise RuntimeError('Generation failed; training not started')
            if generation['state']=='complete':
                break
            time.sleep(30)
        else:
            raise TimeoutError('Generation wait exceeded six hours')
        status['state']='auditing_and_launching'
        save()
        command=[sys.executable,'-u',str(Path(__file__).with_name('start_rc_sonnet_continuation.py')),
                 '--benchmark','r1','--source',str(source)]
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (source/'phase2_launch_out.log').write_text(result.stdout,encoding='utf-8')
        (source/'phase2_launch_err.log').write_text(result.stderr,encoding='utf-8')
        result.check_returncode()
        status['state']='launched'
    except Exception as exc:
        status.update(state='blocked',error=str(exc))
        raise
    finally:
        save()


if __name__=='__main__':
    main()
