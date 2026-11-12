import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location('batch', Path(__file__).resolve().parents[1]/'scripts/swe_verified_batch.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class BatchTests(unittest.TestCase):
    def test_harness_container_limits(self):
        path=Path(__file__).resolve().parents[1]/'scripts/swe_official_grade.py'
        tree=ast.parse(path.read_text())
        classes=[node for node in tree.body if isinstance(node, ast.ClassDef)]
        namespace={}
        exec(compile(ast.Module(body=classes, type_ignores=[]),str(path),'exec'),namespace)
        real=Mock()
        client=namespace['LimitedClient'](real)
        client.containers.create(image='pinned',network_mode='bridge')
        args=real.containers.create.call_args.kwargs
        self.assertEqual(args['network_mode'],'none')
        self.assertEqual(args['mem_limit'],'8g')
        self.assertEqual(args['nano_cpus'],2_000_000_000)
        self.assertIs(client.images,real.images)

    def test_context_terminal_recovery_no_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            slot=Path(tmp)
            trajectory={'info':{'exit_status':'ContextWindowExceededError','submission':''},
                        'messages':[{'role':'exit','extra':{'exit_status':'ContextWindowExceededError','submission':''}}]}
            b.save(slot/'agent/a.traj.json',trajectory)
            b.save(slot/'agent/run.json',{'task_id':'a','mode':'baseline','config':{'model':{'model_name':'test'}}})
            self.assertTrue(b.recover_context_exhaustion(slot,'a'))
            self.assertEqual(b.read(slot/'agent/a.traj.json'),trajectory)
            self.assertEqual(b.read(slot/'agent/prediction.json')['model_patch'],'')

    def test_transport_error_not_converted_to_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            slot=Path(tmp)
            b.save(slot/'agent/a.traj.json',{'info':{'exit_status':'APIConnectionError'},'messages':[{'role':'exit'}]})
            self.assertFalse(b.recover_context_exhaustion(slot,'a'))
            self.assertFalse((slot/'agent/prediction.json').exists())

    def test_image_mapping(self):
        self.assertEqual(b.image_for('psf__requests-2317'), 'swebench/sweb.eval.x86_64.psf_1776_requests-2317:latest')

    def test_partial_summary_no_final_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            b.save(out/'tasks/a/done.json', dict(instance_id='a', resolved=True, empty_patch=False))
            b.save(out/'tasks/b/error.json', {'error':'service down'})
            summary=b.summarize(out, {'ids':['a','b'], 'pilot':['a']})
            self.assertEqual(summary['resolved'],1)
            self.assertIsNone(summary['final_resolved_rate'])
            self.assertEqual(summary['infrastructure_or_unclassified_errors'],['b'])
            self.assertEqual(summary['heldout_completed'],0)

    def test_protocol_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            b.save(out/'inventory.json', {'protocol_hash':'different'})
            with self.assertRaisesRegex(RuntimeError,'Protocol'):
                b.init_data(out)

    def test_empty_prediction_resume_no_model_or_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); slot=out/'tasks/a'
            b.save(slot/'agent/prediction.json', {'instance_id':'a','model_patch':''})
            b.save(slot/'agent/a.traj.json', {'info':{'exit_status':'Submitted'}})
            with patch.object(b,'image_info',return_value={'Id':'sha256:a'}), patch.object(b,'execute') as execute, patch.object(b,'cleanup_image'):
                b.process({'instance_id':'a','problem_statement':'issue','patch':'SECRET'},out)
                execute.assert_not_called()
            self.assertFalse(b.read(slot/'done.json')['resolved'])
            self.assertNotIn('patch',b.read(slot/'instance.json'))
            self.assertNotIn('SECRET',(slot/'instance.json').read_text())

    def test_score_resume_reuses_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); slot=out/'tasks/a'
            b.save(slot/'agent/prediction.json', {'instance_id':'a','model_patch':'diff'})
            b.save(slot/'agent/a.traj.json', {'info':{}})
            def execute(cmd,*args,**kwargs):
                self.assertIn('swe_official_grade.py',' '.join(cmd))
                b.save(slot/'grading/result.json',{'resolved':True})
            with patch.object(b,'image_info',return_value={'Id':'sha256:a'}), patch.object(b,'execute',side_effect=execute), patch.object(b,'cleanup_image'):
                b.process({'instance_id':'a','problem_statement':'issue'},out)
            self.assertTrue(b.read(slot/'done.json')['resolved'])

    def test_changed_image_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            b.save(out/'tasks/a/image.json', {'id':'old'})
            with patch.object(b,'image_info',return_value={'Id':'new'}):
                with self.assertRaisesRegex(RuntimeError,'Image changed'):
                    b.process({'instance_id':'a'},out)

    def test_never_cleanup_preexisting_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            slot=Path(tmp)
            b.save(slot/'image.json', {'owned':False,'tag':'existing','id':'x'})
            with patch.object(b,'image_info',return_value={'Id':'x'}), patch.object(b.subprocess,'run') as run:
                b.cleanup_image(slot)
                run.assert_not_called()

if __name__=='__main__': unittest.main()
