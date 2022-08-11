# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.append('../')
import unittest
from test_all_castle import TestCastleAll


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestCastleAll('test_GES_bic_scatter'))
    suite.addTest(TestCastleAll('test_GES_bic_r2'))
    suite.addTest(TestCastleAll('test_GES_bdeu'))

    runner = unittest.TextTestRunner()
    runner.run(suite)