
"""Matrix and linear-operator primitives for the leg-kinematics library.

This module defines:

    * ``LinearOperator``  -- an abstract base class describing any linear map
      ``T: V -> W`` between vector spaces.
    * ``MatrixOperator``  -- a concrete linear operator backed by a dense 2-D
      numpy array, supporting arbitrary shape ``(m, n)`` and the most relevant
      matrix operations (transpose, inverse, spectrum, kernel, etc.).

The design deliberately reuses numpy for all heavy numerical work while
exposing a small, task-focused API that the rest of the library can build on.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Numeric type used throughout the module.
Scalar = Union[int, float, complex, np.number]


class LinearOperator:
    """Abstract base class for a linear operator.

    A linear operator is a mapping ``T: V -> W`` between two vector spaces
    that is compatible with the vector-space structure, i.e. it satisfies::

        T(a * x + b * y) = a * T(x) + b * T(y)    for all scalars a, b
                                                  and vectors x, y

    In finite-dimensional spaces, choosing bases for ``V`` and ``W`` turns any
    linear operator into a matrix (its coordinate representation). That is the
    concrete role played by :class:`MatrixOperator` below.
    """

    def __init__(self) -> None:
        # Storage for the concrete representation; filled in by subclasses.
        self._data: NDArray | None = None

    # ------------------------------------------------------------------
    # Operator interface (subclasses must override these).
    # ------------------------------------------------------------------
    def apply(self, x: ArrayLike) -> NDArray:
        """Apply the operator to ``x`` and return ``T(x)``."""
        raise NotImplementedError("Subclasses must implement `apply`.")

    def adjoint(self) -> "LinearOperator":
        """Return the adjoint (Hermitian-transpose) operator ``T*``."""
        raise NotImplementedError("Subclasses must implement `adjoint`.")

    def inverse(self) -> "LinearOperator":
        """Return the inverse operator ``T^-1`` if it exists."""
        raise NotImplementedError("Subclasses must implement `inverse`.")


class MatrixOperator(LinearOperator):
    """A linear operator represented by a dense 2-D numpy array.

    The array may have arbitrary shape ``(m, n)``. Square operators expose the
    full spectrum/inverse machinery; rectangular operators still support
    transpose, pseudo-inverse, SVD and null-space computation.

    Examples
    --------
    >>> A = MatrixOperator([[1, 2], [3, 4]])
    >>> A.determinant()
    -2.0
    >>> A.inverse()
    MatrixOperator(
    [[-2. ,  1. ],
     [ 1.5, -0.5]]
    )
    """

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------
    def __init__(self, data: ArrayLike) -> None:
        super().__init__()
        # Convert arbitrary array-like input (nested lists, tuples, ndarray...)
        # into a float64 numpy array and require exactly two axes.
        array = np.asarray(data, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(
                f"MatrixOperator requires a 2-D array, got shape {array.shape}."
            )
        self._data = array

    # ------------------------------------------------------------------
    # Basic attributes and conversion back to plain numpy
    # ------------------------------------------------------------------
    @property
    def data(self) -> NDArray:
        """The underlying numpy array (read-only view)."""
        return self._data.view()

    def to_numpy(self) -> NDArray:
        """Return the underlying data as a plain numpy array (a copy)."""
        return self._data.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(rows, cols)`` of the operator."""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """Number of axes (always 2)."""
        return self._data.ndim

    @property
    def size(self) -> int:
        """Total number of entries."""
        return self._data.size

    def __len__(self) -> int:
        """Number of rows of the operator."""
        return self._data.shape[0]

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"MatrixOperator(\n{self._data!r}\n)"

    def __str__(self) -> str:
        """Human-readable representation (the matrix itself)."""
        return np.array2string(self._data)

    # ------------------------------------------------------------------
    # Helper: coerce operands to a 2-D numpy array
    # ------------------------------------------------------------------
    def _coerce(self, other: ArrayLike | "MatrixOperator") -> NDArray:
        """Return ``other`` as a 2-D numpy array.

        Accepts another ``MatrixOperator`` or any 2-D array-like object.
        """
        if isinstance(other, MatrixOperator):
            return other._data
        array = np.asarray(other, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Operands of MatrixOperator must be 2-D arrays.")
        return array

    # ------------------------------------------------------------------
    # Arithmetic (operator algebra)
    # ------------------------------------------------------------------
    def __add__(self, other: ArrayLike | "MatrixOperator") -> "MatrixOperator":
        """Element-wise addition: ``A + B``."""
        return MatrixOperator(self._data + self._coerce(other))

    def __sub__(self, other: ArrayLike | "MatrixOperator") -> "MatrixOperator":
        """Element-wise subtraction: ``A - B``."""
        return MatrixOperator(self._data - self._coerce(other))

    def __neg__(self) -> "MatrixOperator":
        """Unary negation: ``-A``."""
        return MatrixOperator(-self._data)

    def __mul__(self, scalar: Scalar) -> "MatrixOperator":
        """Scalar multiplication: ``A * c`` (element-wise scaling)."""
        return MatrixOperator(self._data * scalar)

    def __rmul__(self, scalar: Scalar) -> "MatrixOperator":
        """Reflected scalar multiplication: ``c * A``."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Scalar) -> "MatrixOperator":
        """Scalar division: ``A / c``."""
        return MatrixOperator(self._data / scalar)

    def __matmul__(self, other: ArrayLike | "MatrixOperator") -> "MatrixOperator":
        """Matrix product: ``A @ B``."""
        return MatrixOperator(self._data @ self._coerce(other))

    def __pow__(self, power: int) -> "MatrixOperator":
        """Integer matrix power: ``A ** n`` (via repeated squaring)."""
        if not isinstance(power, int):
            raise TypeError("Matrix power must be an integer.")
        return MatrixOperator(np.linalg.matrix_power(self._data, power))

    # ------------------------------------------------------------------
    # Matrix properties: transpose / adjoint
    # ------------------------------------------------------------------
    @property
    def T(self) -> "MatrixOperator":
        """Transpose ``A^T``."""
        return MatrixOperator(self._data.T)

    @property
    def H(self) -> "MatrixOperator":
        """Conjugate (Hermitian) transpose ``A^H``."""
        return MatrixOperator(self._data.conj().T)

    # ------------------------------------------------------------------
    # Implementation of the LinearOperator interface
    # ------------------------------------------------------------------
    def apply(self, x: ArrayLike) -> NDArray:
        """Return ``A @ x`` -- the action of the operator on ``x``."""
        return self._data @ np.asarray(x, dtype=np.float64)

    def adjoint(self) -> "MatrixOperator":
        """Return the adjoint operator ``A^H``."""
        return self.H

    def inverse(self) -> "MatrixOperator":
        """Return the inverse operator ``A^-1``.

        Requires a square, non-singular matrix; raises ``numpy.linalg.LinAlgError``
        otherwise.
        """
        return MatrixOperator(np.linalg.inv(self._data))

    def pinv(self) -> "MatrixOperator":
        """Moore-Penrose pseudo-inverse ``A^+`` (works for rectangular matrices)."""
        return MatrixOperator(np.linalg.pinv(self._data))

    # ------------------------------------------------------------------
    # Spectral properties: spectrum, determinant, trace, rank
    # ------------------------------------------------------------------
    def spectrum(self) -> NDArray:
        """Eigenvalues of the operator (requires a square matrix)."""
        return np.linalg.eigvals(self._data)

    def eigen(self) -> tuple[NDArray, NDArray]:
        """Eigendecomposition: return ``(eigenvalues, eigenvectors)``."""
        return np.linalg.eig(self._data)

    def determinant(self) -> float:
        """Determinant ``det(A)`` (requires a square matrix)."""
        return float(np.linalg.det(self._data))

    def trace(self) -> float:
        """Trace ``tr(A)``."""
        return float(np.trace(self._data))

    def rank(self, tol: float | None = None) -> int:
        """Matrix rank, estimated from singular values."""
        return int(np.linalg.matrix_rank(self._data, tol=tol))

    def norm(self, ord: str | int | None = None) -> float:
        """Matrix norm; the default is the Frobenius norm."""
        return float(np.linalg.norm(self._data, ord=ord))

    def condition_number(self) -> float:
        """Condition number of the operator with respect to the 2-norm."""
        return float(np.linalg.cond(self._data))

    # ------------------------------------------------------------------
    # Subspaces and factorisations
    # ------------------------------------------------------------------
    def kernel(self, tol: float = 1e-12) -> "MatrixOperator":
        """Orthonormal basis of the null space ``ker(A) = {x : A x = 0}``.

        Uses the SVD: the right singular vectors whose singular values are
        (numerically) zero span the null space.
        """
        _, singular_values, vh = np.linalg.svd(self._data, full_matrices=True)
        # Columns of V (rows of Vh) with a near-zero singular value form the basis.
        null_mask = singular_values <= tol
        return MatrixOperator(vh[null_mask].T)

    def image(self, tol: float = 1e-12) -> "MatrixOperator":
        """Orthonormal basis of the column space (range) of the operator."""
        u, singular_values, _ = np.linalg.svd(self._data, full_matrices=True)
        rank = int(np.sum(singular_values > tol))
        return MatrixOperator(u[:, :rank])

    def svd(self) -> tuple[NDArray, NDArray, NDArray]:
        """Singular-value decomposition: return ``(U, S, Vh)``."""
        return np.linalg.svd(self._data, full_matrices=True)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @staticmethod
    def identity(n: int) -> "MatrixOperator":
        """Identity operator of size ``n x n``."""
        return MatrixOperator(np.eye(n))

    @staticmethod
    def zeros(m: int, n: int) -> "MatrixOperator":
        """Zero operator of shape ``m x n``."""
        return MatrixOperator(np.zeros((m, n)))

    @staticmethod
    def ones(m: int, n: int) -> "MatrixOperator":
        """Operator of all ones, shape ``m x n``."""
        return MatrixOperator(np.ones((m, n)))

    @staticmethod
    def diag(values: ArrayLike) -> "MatrixOperator":
        """Diagonal operator with the 1-D array ``values`` on the diagonal."""
        return MatrixOperator(np.diag(np.asarray(values, dtype=np.float64)))