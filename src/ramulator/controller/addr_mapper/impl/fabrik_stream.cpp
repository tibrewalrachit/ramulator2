#include "ramulator/controller/addr_mapper/addr_mapper_base.h"
#include "ramulator/controller/addr_mapper/i_addr_mapper.h"
#include "ramulator/dram/dram_spec.h"

namespace Ramulator {

// Fabrik3D streaming mapper: Row-Bank-Column (MSB to LSB), mixed-radix so
// non-power-of-2 bank counts (e.g. 3 banks/channel) work.
//
// Sequential flit addresses walk the columns of an open row; crossing a row
// boundary moves to the *next bank* at the same row index, so the next
// bank's ACT overlaps the current bank's column walk and a weight stream
// keeps the channel's data bus at full cadence. Only after all banks are
// visited does the row index advance.
class FabrikStream : public IAddrMapper, public AddrMapperBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IAddrMapper, FabrikStream, AddrMapperBase, "FabrikStream")

 private:
  int m_bank_idx = -1;
  Addr_t m_num_banks = -1;
  Addr_t m_num_rows = -1;
  Addr_t m_num_cols = -1;

 public:
  void init() override {
    AddrMapperBase::init();
    const auto& spec = *m_ctrl->m_device.m_spec;
    const auto& level_sizes = spec.organization.level_sizes;
    m_bank_idx = spec.get_level_id("Bank") - 1;
    m_num_banks = level_sizes[spec.get_level_id("Bank")];
    m_num_rows = level_sizes[spec.get_level_id("Row")];
    m_num_cols = level_sizes[spec.level_count - 1] / spec.internal_prefetch_size;
  }

  void apply(Request& req) override {
    req.addr_vec.resize(m_num_mapped_levels + 1, -1);
    Addr_t addr = req.intra_channel_addr >> m_tx_offset;
    req.addr_vec[m_col_idx + 1] = static_cast<int>(addr % m_num_cols);
    addr /= m_num_cols;
    req.addr_vec[m_bank_idx + 1] = static_cast<int>(addr % m_num_banks);
    addr /= m_num_banks;
    req.addr_vec[m_row_idx + 1] = static_cast<int>(addr % m_num_rows);
  }
};

}  // namespace Ramulator
