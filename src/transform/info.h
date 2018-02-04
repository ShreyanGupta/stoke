// Copyright 2013-2016 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STOKE_SRC_TRANSFORM_TRANSFORM_INFO_H
#define STOKE_SRC_TRANSFORM_TRANSFORM_INFO_H

#include "src/ext/x64asm/include/x64asm.h"

namespace stoke {

struct TransformInfo {

  TransformInfo() : success(false), undo_instr(x64asm::NOP), redo_instr(x64asm::NOP) { }

  // Did the transform succeed?
  bool success;

  // Records the type of move performed
  size_t move_type;

  // Records information to undo the transform
  size_t undo_index[2];
  x64asm::Instruction undo_instr;

  // Records information to redo the transform
  x64asm::Instruction redo_instr;
};

} // namespace stoke

#endif
