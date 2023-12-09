## Description

This directory contains test scripts to check if the source code for abstraction construction, DFA Construction, and strategy synthesis, is functioning correctly or not.
- [x] `test script(s) have been implemented`, and
- [ ] means test scripts will be implemented in future iterations.

Here is a list of available tests:

- [x] `LTLf to DFA Construction`
- [ ] `LTL to DFA Construction`
- [ ] `Product Construction w LTLf, LTL, and PDFA formulas`
- [ ] `PDFA Construction`
- [ ] `Transition System Construction`
- [ ] `Two player turn-based game construction`
- [ ] `Adversarial Game (Qualitative Algo.) Strategy Synthesis with 2-player turn-based games w LTL, LTLf tasks`
- [ ] `Cooperative Game (Qualitative Algo.) Strategy Synthesis with 2-player turn-based games w LTL, LTLf tasks`
- [ ] `Qualitative Best-Effort Strategy Synthesis with 2-player turn-based games w Realizable Tasks w LTL, LTLf tasks`
- [ ] `Qualitative Best-Effort Strategy Synthesis with 2-player turn-based games w/o Realizable Tasks w LTL, LTLf tasks`
- [ ] `Quantitative Best-Effort Strategy Synthesis with 2-player turn-based games w Realizable Tasks w LTL, LTLf tasks`
- [ ] `Quantitative Best-Effort Strategy Synthesis with 2-player turn-based games w/o Realizable Tasks w LTL, LTLf tasks`
- [ ] `Qualitative Safe Reach Best-Effort Strategy Synthesis with 2-player turn-based games w Realizable Tasks w LTL, LTLf tasks`
- [ ] `Qualitative Safe Reach Best-Effort Strategy Synthesis with 2-player turn-based games w/o Realizable Tasks w LTL, LTLf tasks`
- [ ] `Value Iteration (Min-Max) Strategy Synthesis with 2-player turn-based games w LTL, LTLf, PDFA tasks`
- [ ] `Value Iteration (Min-Min) Strategy Synthesis with 2-player turn-based games w LTL, LTLf, PDFA tasks`
- [ ] `Finite Trace Regret Strategy Synthesis (ICRA 22) with 2-player turn-based games`

### Test Packages

To run each test package, use the following command

```bash
python3 -m unittest discover -s tests.<directory-name> -bv
```

The `-s` flag allows you to specify directory to start discovery from. Use only `-b` if you want to suppress all the prints (including progress). Use `-v` for verbose print or `-bv` to just print if a test failed or pass.


### Test Scripts

To run the test scripts within each package, use the following command to run one module

```bash
cd <root/of/project>

python3 -m tests.<directory-name>.<module-nane> -b
```

The `-m` flag runs the test as a module while `-b` flag suppresses the output of the code if all the tests pass (you can also use `-bv`). If the test fails, then it throws the error message at the very beginning and then the rest of the output. 

### Test everything

To run all the tests use the following command

```bash
python3 -m unittest -bv
```
