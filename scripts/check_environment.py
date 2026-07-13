from snn_space.utils.seed import set_seed
from snn_space.utils.device import get_device
from snn_space.utils.logger import get_logger

logger = get_logger(__name__)

set_seed(42)

device = get_device()

logger.info(f"Device: {device}")

logger.info("Everything is ready 🚀")
