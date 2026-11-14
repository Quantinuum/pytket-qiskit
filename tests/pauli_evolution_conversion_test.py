# Copyright Quantinuum
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

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister  # type: ignore[import-untyped]
from qiskit.circuit.library import PauliEvolutionGate  # type: ignore[import-untyped]
from qiskit.quantum_info import Operator, SparsePauliOp  # type: ignore[import-untyped]

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import DecomposeBoxes


@pytest.mark.parametrize("label", ["X", "IZ", "ZI", "XY", "IXZ", "III"])
@pytest.mark.parametrize("time", [0.7, -0.3])
@pytest.mark.parametrize("reverse_qargs", [False, True])
def test_pauli_evolution_preserves_unitary(
    label: str, time: float, reverse_qargs: bool
) -> None:
    circuit = QuantumCircuit(len(label))
    qargs = list(range(len(label)))
    if reverse_qargs:
        qargs.reverse()
    gate = PauliEvolutionGate(SparsePauliOp([label], coeffs=[0.6]), time=time)
    circuit.append(gate, qargs)

    expected = Operator(circuit).data
    converted = qiskit_to_tk(circuit)
    DecomposeBoxes().apply(converted)
    actual = Operator(tk_to_qiskit(converted)).data
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_commuting_pauli_evolution_preserves_unitary() -> None:
    circuit = QuantumCircuit(2)
    operator = SparsePauliOp(["ZI", "IZ", "II"], coeffs=[0.7, -0.4, 0.2])
    circuit.append(PauliEvolutionGate(operator, time=0.3), [0, 1])
    expected = Operator(circuit).data
    converted = qiskit_to_tk(circuit)
    DecomposeBoxes().apply(converted)
    actual = Operator(tk_to_qiskit(converted)).data
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("qargs", [[1, 3], [3, 1]])
def test_pauli_evolution_on_register_subset(qargs: list[int]) -> None:
    circuit = QuantumCircuit(QuantumRegister(4, "data"))
    circuit.append(PauliEvolutionGate(SparsePauliOp("XY"), time=0.4), qargs)
    expected = Operator(circuit).data
    converted = qiskit_to_tk(circuit)
    DecomposeBoxes().apply(converted)
    actual = Operator(tk_to_qiskit(converted)).data
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
