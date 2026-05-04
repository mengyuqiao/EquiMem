class MemoryBase(ABC):
    """Interface that calibration expects."""

    @abstractmethod
    def save(self, content: str, trust_weight: float = 1.0) -> None:
        """Write an entry to memory."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k relevant entries."""

    @abstractmethod
    def search_with_delta(self, query: str, delta, top_k: int) -> list[str]:
        """Simulate retrieval after merging delta (for rho_align)."""

    @abstractmethod
    def get_state(self) -> Any:
        """Return current memory state for calibration."""