"""Run the complete automated suite and create a factual regression report."""
import json, subprocess, sys, xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
report_dir = ROOT/'reports'/'day3'
report_dir.mkdir(parents=True, exist_ok=True)
xmlpath = report_dir/'latest_pytest_junit.xml'
cmd = [sys.executable,'-m','pytest','tests','-q',f'--junitxml={xmlpath}']
run = subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(run.stdout)
(report_dir/'latest_pytest_output.txt').write_text(run.stdout,encoding='utf-8')
summary = {'total':0,'passed':0,'failed':0,'errors':0,'skipped':0,'pass_percentage':0}
if xmlpath.exists():
    suites=ET.parse(xmlpath).getroot()
    cases=suites.findall('.//testcase')
    summary['total']=len(cases)
    summary['failed']=sum(c.find('failure') is not None for c in cases)
    summary['errors']=sum(c.find('error') is not None for c in cases)
    summary['skipped']=sum(c.find('skipped') is not None for c in cases)
    summary['passed']=summary['total']-summary['failed']-summary['errors']-summary['skipped']
    summary['pass_percentage']=round(100*summary['passed']/summary['total'],2) if summary['total'] else 0
report={'generated_at':datetime.now(timezone.utc).isoformat(),'execution_environment':'local pytest with synthetic data and offline LLM fallback','command':' '.join(cmd),'returncode':run.returncode,'regression_summary':summary,'release_decision':'NO-GO' if run.returncode or summary['failed'] or summary['errors'] else 'CONDITIONAL GO','limitations':['Offline fallback is not evidence of a live LLM evaluation','TestClient concurrency is not production-scale load testing','Pre-existing artifacts are not independently retrained or fully audited','Release gates require stakeholder approval']}
(report_dir/'latest_regression_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
sys.exit(run.returncode)
