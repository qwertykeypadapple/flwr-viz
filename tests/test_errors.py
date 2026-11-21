from flwr_viz.errors import short_exception, short_reason

RAY_REASON = (
    "<class 'ray.exceptions.RayTaskError(ClientAppException)'>:<'ray::ClientAppActor.run() (pid=1, ip=127.0.0.1)\n"
    "  File \"/x/arrow_writer.py\", line 765, in write_table\n"
    "OSError: [Errno 28] No space left on device\n\n"
    "The above exception was the direct cause of the following exception:\n\n"
    "datasets.exceptions.DatasetGenerationError: An error occurred while generating the dataset\n"
    "flwr.clientapp.client_app.ClientAppException: \n'>"
)


def test_reason_picks_root_cause_and_drops_ray_noise():
    out = short_reason(RAY_REASON)
    assert out == ("DatasetGenerationError: An error occurred while generating the dataset "
                   "(caused by OSError: [Errno 28] No space left on device)")
    assert "/x/arrow_writer.py" not in out          # no traceback paths


def test_reason_without_exception_lines():
    assert short_reason("node went away") == "node went away"
    assert short_reason("") == ""


def test_exception_chain():
    try:
        try:
            raise OSError(28, "No space left on device")
        except OSError as e:
            raise RuntimeError("could not build dataset\nsecond line") from e
    except RuntimeError as e:
        assert short_exception(e) == ("RuntimeError: could not build dataset "
                                      "(caused by OSError: [Errno 28] No space left on device)")
    assert short_exception(ValueError("plain")) == "ValueError: plain"
