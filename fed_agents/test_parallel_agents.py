from fed_agents.parallel_orchestrator import run_parallel_agents


def test_parallel_agents_run_successfully():
    result = run_parallel_agents()
    assert 'fed_summary' in result
    assert 'research_result' in result
    assert result['research_result']['status'] == 'report_ready'
    assert 'Best model' in result['research_result']['report']
