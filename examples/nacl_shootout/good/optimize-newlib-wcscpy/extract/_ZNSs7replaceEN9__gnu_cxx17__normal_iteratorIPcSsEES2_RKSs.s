  .text
  .globl _ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_RKSs
  .type _ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_RKSs, @function

#! file-offset 0xef320
#! rip-offset  0xaf320
#! capacity    32 bytes

# Text                                                        #  Line  RIP      Bytes  Opcode            
._ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_RKSs:  #        0xaf320  0      OPC=<label>       
  movl %ecx, %ecx                                             #  1     0xaf320  2      OPC=movl_r32_r32  
  movl %edi, %edi                                             #  2     0xaf322  2      OPC=movl_r32_r32  
  subl %esi, %edx                                             #  3     0xaf324  2      OPC=subl_r32_r32  
  movl %ecx, %ecx                                             #  4     0xaf326  2      OPC=movl_r32_r32  
  movl (%r15,%rcx,1), %ecx                                    #  5     0xaf328  4      OPC=movl_r32_m32  
  movl %edi, %edi                                             #  6     0xaf32c  2      OPC=movl_r32_r32  
  subl (%r15,%rdi,1), %esi                                    #  7     0xaf32e  4      OPC=subl_r32_m32  
  leal -0xc(%rcx), %eax                                       #  8     0xaf332  3      OPC=leal_r32_m16  
  movl %eax, %eax                                             #  9     0xaf335  2      OPC=movl_r32_r32  
  movl (%r15,%rax,1), %r8d                                    #  10    0xaf337  4      OPC=movl_r32_m32  
  jmpq ._ZNSs7replaceEjjPKcj                                  #  11    0xaf33b  5      OPC=jmpq_label_1  
                                                                                                         
.size _ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_RKSs, .-_ZNSs7replaceEN9__gnu_cxx17__normal_iteratorIPcSsEES2_RKSs

