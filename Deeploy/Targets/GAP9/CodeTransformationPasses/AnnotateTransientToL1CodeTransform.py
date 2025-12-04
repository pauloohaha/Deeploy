# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t
from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    TransientBuffer, _NoVerbosity, NodeTemplate



class AnnotateTransientBuffersToL1CodeTransform(CodeTransformationPass):
    """
    Code transformation pass to annotate transient (intermediate) buffers to L1 memory.

    This is a CodeTransformationPass version (not OptimizationPass), which means it can be
    used directly in CodeTransformation pipelines like GAP9Transformer.

    This pass is designed to work with the GAP9 memory hierarchy where:
    - L2: 256KB external shared RAM
    - L1: 64KB cluster-local fast RAM

    Transient buffers are temporary scratch buffers used by kernels (e.g., im2col
    buffers in convolutions). By placing them in L1, we avoid DMA overhead.

    Usage in Transformer:
    ---------------------
    GAP9L2OnlyTransformerL1Transient = CodeTransformation([
        AnnotateTransientBuffersToL1CodeTransform("L1"),  # Annotate transients to L1
        MemoryManagementGeneration("L1"),                 # Manage L1 memory
        TilingVariableReplacement("L2"),
        # ... rest of pipeline
    ])

    Notes:
    ------
    - This pass runs during code transformation (after binding)
    - It respects already-annotated buffers (won't override by default)
    - Safe to use with tiling transformations
    - Must be placed BEFORE MemoryManagementGeneration passes
    """

    def __init__(self, targetMemoryLevel: str = "L1", overrideExisting: bool = False):
        """
        Parameters
        ----------
        targetMemoryLevel : str, optional
            Memory level to assign to transient buffers (default: "L1")
        overrideExisting : bool, optional
            Whether to override already-annotated buffers (default: False)
            Setting to False is safer and respects earlier annotations
        """
        super().__init__()
        self.targetMemoryLevel = targetMemoryLevel
        self.overrideExisting = overrideExisting
        self.currentOffset = 0  # Track current arena offset
        self.arenaName = f"MEMORYARENA_{targetMemoryLevel}"

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Apply memory level annotation to all transient buffers in the current context.

        This method is called for each ExecutionBlock during code transformation.

        Parameters
        ----------
        ctxt : NetworkContext
            Current network context containing all buffers
        executionBlock : ExecutionBlock
            Current execution block being transformed (not modified by this pass)
        name : str
            Name of the current node/block
        verbose : CodeGenVerbosity, optional
            Verbosity level for code generation

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            Updated context and unchanged execution block
        """
        # Create or get the arena buffer
        if not ctxt.is_global(self.arenaName):
            raise("L1 arena not found")

        arena = ctxt.lookup(self.arenaName)
        currentOffset = 0 #reset arena offset for each time of transform

        # Iterate through all buffers in both local and global contexts
        for _buffer in {**ctxt.localObjects, **ctxt.globalObjects}.values():
            # Check if it's a transient buffer
            if isinstance(_buffer, TransientBuffer) and _buffer._users[0] == name:#only transform for trasient buffer in current layer
                # Only annotate if:
                # 1. Buffer doesn't have _memoryLevel yet, OR
                # 2. overrideExisting is True
                if not hasattr(_buffer, "_memoryLevel") or self.overrideExisting:
                    _buffer._memoryLevel = self.targetMemoryLevel

                    # Calculate buffer size in bytes
                    bufferSize = _buffer.size  # size is already in bytes for TransientBuffer

                    # Allocate from arena at current offset
                    _buffer.initTemplate = NodeTemplate("")  # Disable init template
                    _buffer.allocTemplate = NodeTemplate(
                        "${type.typeName} ${name} = (${type.typeName}) " +
                        f"((char*){str(arena._instance)} + {currentOffset});")
                    _buffer.deallocTemplate = NodeTemplate("")

                    # Update offset for next buffer
                    currentOffset += bufferSize
            

        return ctxt, executionBlock
