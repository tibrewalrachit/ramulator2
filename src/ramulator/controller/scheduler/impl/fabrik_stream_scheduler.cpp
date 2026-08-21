#include <vector>

#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/scheduler/i_scheduler.h"

namespace Ramulator {

// Stream-ahead scheduler for streaming-dominated (Fabrik3D-style) channels.
//
// Plain FRFCFS starves row commands under a pure stream: a ready row-hit RD
// exists on every cycle, so the ACT/PRE for the next row is only issued once
// the current row is drained, exposing nRP+nRCD at every row crossing
// (~73% of peak at 700 MHz with 32-flit rows).
//
// This scheduler spends one command slot early instead: if any queued
// request needs a row command (ACT or PRE) on a bank that has *no* pending
// row-hits — i.e. the stream has moved on or not yet arrived — and that
// command is timing-ready, issue it ahead of the ready RDs. The next row is
// then open by the time the stream reaches it, and steady state becomes
// (cols + 2) command slots per row instead of (cols + nRP + nRCD) cycles.
// Requests to banks that still have row-hits pending are never preempted,
// so mixed traffic degrades to FRFCFS behavior.
class FabrikStreamScheduler : public IScheduler, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IScheduler, FabrikStreamScheduler, "StreamAhead")

  ControllerBase* m_ctrl = nullptr;
  std::vector<bool> m_bank_has_rowhit;

  void init() override {
    m_ctrl = cast_parent<ControllerBase>();
    m_bank_has_rowhit.assign(m_ctrl->m_device.m_bank_nodes.size(), false);
  }

  ReqBuffer::iterator get_best_request(ReqBuffer& buffer, RequestFilterRef filter) override {
    if (buffer.size() == 0) {
      return buffer.end();
    }

    std::fill(m_bank_has_rowhit.begin(), m_bank_has_rowhit.end(), false);

    // Pass 1: resolve prerequisite commands; mark banks with pending hits.
    for (auto it = buffer.begin(); it != buffer.end(); it++) {
      it->command = m_ctrl->get_preq_command(it->final_command, it->addr_vec);
      if (m_ctrl->m_device.check_rowbuffer_hit(it->final_command, it->addr_vec, m_ctrl->m_clk)) {
        const int bank_id = m_ctrl->m_device.get_flat_bank_id(it->addr_vec);
        if (bank_id >= 0 && bank_id < static_cast<int>(m_bank_has_rowhit.size())) {
          m_bank_has_rowhit[bank_id] = true;
        }
      }
    }

    // Pass 2: oldest timing-ready row command on a hit-free bank.
    auto prep = buffer.end();
    for (auto it = buffer.begin(); it != buffer.end(); it++) {
      if (filter && !filter(*it)) {
        continue;
      }
      if (it->command == it->final_command) {
        continue;  // column command, not row preparation
      }
      const int bank_id = m_ctrl->m_device.get_flat_bank_id(it->addr_vec);
      if (bank_id >= 0 && bank_id < static_cast<int>(m_bank_has_rowhit.size()) && m_bank_has_rowhit[bank_id]) {
        continue;  // bank is still streaming an open row — don't preempt
      }
      if (!m_ctrl->check_timing(it->command, it->addr_vec)) {
        continue;
      }
      if (prep == buffer.end() || it->arrive < prep->arrive) {
        prep = it;
      }
    }
    if (prep != buffer.end()) {
      return prep;
    }

    // Pass 3: plain FRFCFS.
    auto candidate = buffer.end();
    bool cand_timing_ok = false;
    for (auto it = buffer.begin(); it != buffer.end(); it++) {
      if (filter && !filter(*it)) {
        continue;
      }
      if (candidate == buffer.end()) {
        candidate = it;
        cand_timing_ok = m_ctrl->check_timing(it->command, it->addr_vec);
        continue;
      }
      bool it_timing_ok = m_ctrl->check_timing(it->command, it->addr_vec);
      if (cand_timing_ok != it_timing_ok) {
        if (it_timing_ok) {
          candidate = it;
          cand_timing_ok = true;
        }
      } else if (it->arrive < candidate->arrive) {
        candidate = it;
      }
    }
    return candidate;
  }
};

}  // namespace Ramulator
