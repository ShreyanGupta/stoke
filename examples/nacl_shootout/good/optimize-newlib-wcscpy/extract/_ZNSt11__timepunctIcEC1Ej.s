  .text
  .globl _ZNSt11__timepunctIcEC1Ej
  .type _ZNSt11__timepunctIcEC1Ej, @function

#! file-offset 0xbd580
#! rip-offset  0x7d580
#! capacity    192 bytes

# Text                                                      #  Line  RIP      Bytes  Opcode              
._ZNSt11__timepunctIcEC1Ej:                                 #        0x7d580  0      OPC=<label>         
  pushq %rbx                                                #  1     0x7d580  1      OPC=pushq_r64_1     
  xorl %eax, %eax                                           #  2     0x7d581  2      OPC=xorl_r32_r32    
  movl %edi, %ebx                                           #  3     0x7d583  2      OPC=movl_r32_r32    
  subl $0x10, %esp                                          #  4     0x7d585  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                                           #  5     0x7d588  3      OPC=addq_r64_r64    
  testl %esi, %esi                                          #  6     0x7d58b  2      OPC=testl_r32_r32   
  movl %ebx, %ebx                                           #  7     0x7d58d  2      OPC=movl_r32_r32    
  movl $0x1003aec8, (%r15,%rbx,1)                           #  8     0x7d58f  8      OPC=movl_m32_imm32  
  setne %al                                                 #  9     0x7d597  3      OPC=setne_r8        
  nop                                                       #  10    0x7d59a  1      OPC=nop             
  nop                                                       #  11    0x7d59b  1      OPC=nop             
  nop                                                       #  12    0x7d59c  1      OPC=nop             
  nop                                                       #  13    0x7d59d  1      OPC=nop             
  nop                                                       #  14    0x7d59e  1      OPC=nop             
  nop                                                       #  15    0x7d59f  1      OPC=nop             
  movl %ebx, %ebx                                           #  16    0x7d5a0  2      OPC=movl_r32_r32    
  movl $0x0, 0x8(%r15,%rbx,1)                               #  17    0x7d5a2  9      OPC=movl_m32_imm32  
  movl %ebx, %ebx                                           #  18    0x7d5ab  2      OPC=movl_r32_r32    
  movl %eax, 0x4(%r15,%rbx,1)                               #  19    0x7d5ad  5      OPC=movl_m32_r32    
  nop                                                       #  20    0x7d5b2  1      OPC=nop             
  nop                                                       #  21    0x7d5b3  1      OPC=nop             
  nop                                                       #  22    0x7d5b4  1      OPC=nop             
  nop                                                       #  23    0x7d5b5  1      OPC=nop             
  nop                                                       #  24    0x7d5b6  1      OPC=nop             
  nop                                                       #  25    0x7d5b7  1      OPC=nop             
  nop                                                       #  26    0x7d5b8  1      OPC=nop             
  nop                                                       #  27    0x7d5b9  1      OPC=nop             
  nop                                                       #  28    0x7d5ba  1      OPC=nop             
  callq ._ZNSt6locale5facet13_S_get_c_nameEv                #  29    0x7d5bb  5      OPC=callq_label     
  movl %ebx, %ebx                                           #  30    0x7d5c0  2      OPC=movl_r32_r32    
  movl %eax, 0x10(%r15,%rbx,1)                              #  31    0x7d5c2  5      OPC=movl_m32_r32    
  xorl %esi, %esi                                           #  32    0x7d5c7  2      OPC=xorl_r32_r32    
  movl %ebx, %edi                                           #  33    0x7d5c9  2      OPC=movl_r32_r32    
  nop                                                       #  34    0x7d5cb  1      OPC=nop             
  nop                                                       #  35    0x7d5cc  1      OPC=nop             
  nop                                                       #  36    0x7d5cd  1      OPC=nop             
  nop                                                       #  37    0x7d5ce  1      OPC=nop             
  nop                                                       #  38    0x7d5cf  1      OPC=nop             
  nop                                                       #  39    0x7d5d0  1      OPC=nop             
  nop                                                       #  40    0x7d5d1  1      OPC=nop             
  nop                                                       #  41    0x7d5d2  1      OPC=nop             
  nop                                                       #  42    0x7d5d3  1      OPC=nop             
  nop                                                       #  43    0x7d5d4  1      OPC=nop             
  nop                                                       #  44    0x7d5d5  1      OPC=nop             
  nop                                                       #  45    0x7d5d6  1      OPC=nop             
  nop                                                       #  46    0x7d5d7  1      OPC=nop             
  nop                                                       #  47    0x7d5d8  1      OPC=nop             
  nop                                                       #  48    0x7d5d9  1      OPC=nop             
  nop                                                       #  49    0x7d5da  1      OPC=nop             
  callq ._ZNSt11__timepunctIcE23_M_initialize_timepunctEPi  #  50    0x7d5db  5      OPC=callq_label     
  addl $0x10, %esp                                          #  51    0x7d5e0  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                                           #  52    0x7d5e3  3      OPC=addq_r64_r64    
  popq %rbx                                                 #  53    0x7d5e6  1      OPC=popq_r64_1      
  popq %r11                                                 #  54    0x7d5e7  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d                                   #  55    0x7d5e9  7      OPC=andl_r32_imm32  
  nop                                                       #  56    0x7d5f0  1      OPC=nop             
  nop                                                       #  57    0x7d5f1  1      OPC=nop             
  nop                                                       #  58    0x7d5f2  1      OPC=nop             
  nop                                                       #  59    0x7d5f3  1      OPC=nop             
  addq %r15, %r11                                           #  60    0x7d5f4  3      OPC=addq_r64_r64    
  jmpq %r11                                                 #  61    0x7d5f7  3      OPC=jmpq_r64        
  nop                                                       #  62    0x7d5fa  1      OPC=nop             
  nop                                                       #  63    0x7d5fb  1      OPC=nop             
  nop                                                       #  64    0x7d5fc  1      OPC=nop             
  nop                                                       #  65    0x7d5fd  1      OPC=nop             
  nop                                                       #  66    0x7d5fe  1      OPC=nop             
  nop                                                       #  67    0x7d5ff  1      OPC=nop             
  nop                                                       #  68    0x7d600  1      OPC=nop             
  nop                                                       #  69    0x7d601  1      OPC=nop             
  nop                                                       #  70    0x7d602  1      OPC=nop             
  nop                                                       #  71    0x7d603  1      OPC=nop             
  nop                                                       #  72    0x7d604  1      OPC=nop             
  nop                                                       #  73    0x7d605  1      OPC=nop             
  nop                                                       #  74    0x7d606  1      OPC=nop             
  movl %ebx, %edi                                           #  75    0x7d607  2      OPC=movl_r32_r32    
  movl %eax, 0x8(%rsp)                                      #  76    0x7d609  4      OPC=movl_m32_r32    
  nop                                                       #  77    0x7d60d  1      OPC=nop             
  nop                                                       #  78    0x7d60e  1      OPC=nop             
  nop                                                       #  79    0x7d60f  1      OPC=nop             
  nop                                                       #  80    0x7d610  1      OPC=nop             
  nop                                                       #  81    0x7d611  1      OPC=nop             
  nop                                                       #  82    0x7d612  1      OPC=nop             
  nop                                                       #  83    0x7d613  1      OPC=nop             
  nop                                                       #  84    0x7d614  1      OPC=nop             
  nop                                                       #  85    0x7d615  1      OPC=nop             
  nop                                                       #  86    0x7d616  1      OPC=nop             
  nop                                                       #  87    0x7d617  1      OPC=nop             
  nop                                                       #  88    0x7d618  1      OPC=nop             
  nop                                                       #  89    0x7d619  1      OPC=nop             
  nop                                                       #  90    0x7d61a  1      OPC=nop             
  nop                                                       #  91    0x7d61b  1      OPC=nop             
  nop                                                       #  92    0x7d61c  1      OPC=nop             
  nop                                                       #  93    0x7d61d  1      OPC=nop             
  nop                                                       #  94    0x7d61e  1      OPC=nop             
  nop                                                       #  95    0x7d61f  1      OPC=nop             
  nop                                                       #  96    0x7d620  1      OPC=nop             
  nop                                                       #  97    0x7d621  1      OPC=nop             
  callq ._ZNSt6locale5facetD2Ev                             #  98    0x7d622  5      OPC=callq_label     
  movl 0x8(%rsp), %eax                                      #  99    0x7d627  4      OPC=movl_r32_m32    
  movl %eax, %edi                                           #  100   0x7d62b  2      OPC=movl_r32_r32    
  nop                                                       #  101   0x7d62d  1      OPC=nop             
  nop                                                       #  102   0x7d62e  1      OPC=nop             
  nop                                                       #  103   0x7d62f  1      OPC=nop             
  nop                                                       #  104   0x7d630  1      OPC=nop             
  nop                                                       #  105   0x7d631  1      OPC=nop             
  nop                                                       #  106   0x7d632  1      OPC=nop             
  nop                                                       #  107   0x7d633  1      OPC=nop             
  nop                                                       #  108   0x7d634  1      OPC=nop             
  nop                                                       #  109   0x7d635  1      OPC=nop             
  nop                                                       #  110   0x7d636  1      OPC=nop             
  nop                                                       #  111   0x7d637  1      OPC=nop             
  nop                                                       #  112   0x7d638  1      OPC=nop             
  nop                                                       #  113   0x7d639  1      OPC=nop             
  nop                                                       #  114   0x7d63a  1      OPC=nop             
  nop                                                       #  115   0x7d63b  1      OPC=nop             
  nop                                                       #  116   0x7d63c  1      OPC=nop             
  nop                                                       #  117   0x7d63d  1      OPC=nop             
  nop                                                       #  118   0x7d63e  1      OPC=nop             
  nop                                                       #  119   0x7d63f  1      OPC=nop             
  nop                                                       #  120   0x7d640  1      OPC=nop             
  nop                                                       #  121   0x7d641  1      OPC=nop             
  callq ._Unwind_Resume                                     #  122   0x7d642  5      OPC=callq_label     
                                                                                                         
.size _ZNSt11__timepunctIcEC1Ej, .-_ZNSt11__timepunctIcEC1Ej

