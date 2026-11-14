"""Native diagonal conversion with independent full-operator comparisons."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister  # type: ignore[import-untyped]
from qiskit.circuit.library import DiagonalGate  # type: ignore[import-untyped]
from qiskit.quantum_info import Operator  # type: ignore[import-untyped]

from pytket.circuit import Circuit, DiagonalBox, OpType
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import DecomposeBoxes


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_qiskit_diagonal_conversion(n_qubits: int, subset: bool, inverse: bool) -> None:
    diagonal = np.exp(1j * np.linspace(-0.71, 2.31, 2**n_qubits))
    register = QuantumRegister(n_qubits + 2, "data")
    circuit = QuantumCircuit(register)
    wires = [n_qubits + 1, *range(n_qubits - 1)] if subset else list(range(n_qubits))
    gate = DiagonalGate(diagonal)
    circuit.append(gate.inverse() if inverse else gate, [register[i] for i in wires])

    converted = qiskit_to_tk(circuit)
    assert converted.n_gates_of_type(OpType.DiagonalBox) == 1
    decomposed = converted.copy()
    DecomposeBoxes().apply(decomposed)
    np.testing.assert_allclose(
        decomposed.get_unitary(), Operator(circuit.reverse_bits()).data, atol=1e-10
    )
    roundtrip = tk_to_qiskit(converted)
    assert isinstance(roundtrip.data[0].operation, DiagonalGate)
    np.testing.assert_allclose(
        Operator(roundtrip).data, Operator(circuit).data, atol=1e-10
    )


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
@pytest.mark.parametrize("subset", [False, True])
def test_tket_diagonal_conversion(n_qubits: int, subset: bool) -> None:
    diagonal = np.exp(1j * np.linspace(0.17, -2.11, 2**n_qubits))
    circuit = Circuit()
    register = circuit.add_q_register("data", n_qubits + 2)
    wires = [n_qubits + 1, *range(n_qubits - 1)] if subset else list(range(n_qubits))
    circuit.add_diagonal_box(DiagonalBox(diagonal), [register[i] for i in wires])
    converted = tk_to_qiskit(circuit)
    assert isinstance(converted.data[0].operation, DiagonalGate)
    decomposed = circuit.copy()
    DecomposeBoxes().apply(decomposed)
    np.testing.assert_allclose(
        Operator(converted.reverse_bits()).data, decomposed.get_unitary(), atol=1e-10
    )


def test_conditioned_diagonal_conversion() -> None:
    diagonal = np.exp(1j * np.array([0.1, 0.7, -0.9, 1.3]))
    circuit = Circuit(2, 1)
    circuit.add_diagonal_box(
        DiagonalBox(diagonal), [1, 0], condition_bits=[0], condition_value=1
    )
    converted = tk_to_qiskit(circuit)
    conditional = converted.data[0].operation
    assert conditional.name == "if_else"
    assert conditional.condition[1] == 1
    instruction = conditional.blocks[0].data[0].operation
    assert isinstance(instruction, DiagonalGate)
    np.testing.assert_allclose(
        Operator(instruction).data, np.diag(diagonal), atol=1e-10
    )
