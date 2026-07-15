import os
import tempfile
import unittest
from unittest.mock import call, patch

from paradex.io.camera_system.signal_generator import UTGE900


class UTGE900Tests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp()
        os.close(handle)
        self.generator = UTGE900(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_start_programs_paraoffice_signal_then_enables_output(self):
        with patch.object(self.generator, "write") as write, patch.object(
            self.generator, "_write_verify"
        ) as verify, patch("paradex.io.camera_system.signal_generator.time.sleep"):
            self.generator.start(fps=30)

        self.assertEqual(
            verify.call_args_list,
            [
                call(":CHANnel1:LOAD", 10000),
                call(":CHANnel1:BASE:WAVE", "SQUare"),
                call(":CHANnel1:BASE:AMPLitude", 5.0),
                call(":CHANnel1:BASE:OFFSet", 2.5),
                call(":CHANnel1:BASE:FREQuency", 30.0),
                call(":CHANnel1:BASE:DUTY", 50.0),
            ],
        )
        self.assertEqual(
            write.call_args_list,
            [call(":CHANnel1:OUTPut OFF"), call(":CHANnel1:OUTPut ON")],
        )
        self.assertTrue(self.generator.ch[0])

    def test_stop_uses_scpi_output_off(self):
        self.generator.ch[0] = True
        with patch.object(self.generator, "write") as write:
            self.generator.stop()
        write.assert_called_once_with(":CHANnel1:OUTPut OFF")
        self.assertFalse(self.generator.ch[0])

    def test_write_verify_accepts_abbreviated_string_and_numeric_response(self):
        with patch.object(self.generator, "write"), patch.object(
            self.generator, "query", side_effect=["SQU", "3.000000e+01"]
        ):
            self.generator._write_verify(":CHANnel1:BASE:WAVE", "SQUare")
            self.generator._write_verify(":CHANnel1:BASE:FREQuency", 30.0)


if __name__ == "__main__":
    unittest.main()
