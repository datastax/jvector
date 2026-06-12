### Testing Enhancements

**Description**  
Enhancements to the JVector testing infrastructure:
- On disk index cache added for Grid benchmark harness
- Logging subsystem overhaul
- New JMH tests
- Test results now include metrics for `nodes visited`, `heap usage`, `disk usage`, `PQ Distance`

**Purpose / Impact**
- Faster testing cycle
- Better comprehension of test results
- new metrics to compare inter-release

**Notes**
- Used internally by JVector, no client impact

**Related Issues**
- [615](https://github.com/datastax/jvector/issues/615)
- [616](https://github.com/datastax/jvector/issues/616)

