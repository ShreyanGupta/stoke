  .text
  .globl __gmp_assert_fail
  .type __gmp_assert_fail, @function

#! file-offset 0x779a0
#! rip-offset  0x379a0
#! capacity    160 bytes

# Text                         #  Line  RIP      Bytes  Opcode              
.__gmp_assert_fail:            #        0x379a0  0      OPC=<label>         
  pushq %rbx                   #  1     0x379a0  1      OPC=pushq_r64_1     
  movl %edi, %edi              #  2     0x379a1  2      OPC=movl_r32_r32    
  movl %edx, %ebx              #  3     0x379a3  2      OPC=movl_r32_r32    
  nop                          #  4     0x379a5  1      OPC=nop             
  nop                          #  5     0x379a6  1      OPC=nop             
  nop                          #  6     0x379a7  1      OPC=nop             
  nop                          #  7     0x379a8  1      OPC=nop             
  nop                          #  8     0x379a9  1      OPC=nop             
  nop                          #  9     0x379aa  1      OPC=nop             
  nop                          #  10    0x379ab  1      OPC=nop             
  nop                          #  11    0x379ac  1      OPC=nop             
  nop                          #  12    0x379ad  1      OPC=nop             
  nop                          #  13    0x379ae  1      OPC=nop             
  nop                          #  14    0x379af  1      OPC=nop             
  nop                          #  15    0x379b0  1      OPC=nop             
  nop                          #  16    0x379b1  1      OPC=nop             
  nop                          #  17    0x379b2  1      OPC=nop             
  nop                          #  18    0x379b3  1      OPC=nop             
  nop                          #  19    0x379b4  1      OPC=nop             
  nop                          #  20    0x379b5  1      OPC=nop             
  nop                          #  21    0x379b6  1      OPC=nop             
  nop                          #  22    0x379b7  1      OPC=nop             
  nop                          #  23    0x379b8  1      OPC=nop             
  nop                          #  24    0x379b9  1      OPC=nop             
  nop                          #  25    0x379ba  1      OPC=nop             
  callq .__gmp_assert_header   #  26    0x379bb  5      OPC=callq_label     
  nop                          #  27    0x379c0  1      OPC=nop             
  nop                          #  28    0x379c1  1      OPC=nop             
  nop                          #  29    0x379c2  1      OPC=nop             
  nop                          #  30    0x379c3  1      OPC=nop             
  nop                          #  31    0x379c4  1      OPC=nop             
  nop                          #  32    0x379c5  1      OPC=nop             
  nop                          #  33    0x379c6  1      OPC=nop             
  nop                          #  34    0x379c7  1      OPC=nop             
  nop                          #  35    0x379c8  1      OPC=nop             
  nop                          #  36    0x379c9  1      OPC=nop             
  nop                          #  37    0x379ca  1      OPC=nop             
  nop                          #  38    0x379cb  1      OPC=nop             
  nop                          #  39    0x379cc  1      OPC=nop             
  nop                          #  40    0x379cd  1      OPC=nop             
  nop                          #  41    0x379ce  1      OPC=nop             
  nop                          #  42    0x379cf  1      OPC=nop             
  nop                          #  43    0x379d0  1      OPC=nop             
  nop                          #  44    0x379d1  1      OPC=nop             
  nop                          #  45    0x379d2  1      OPC=nop             
  nop                          #  46    0x379d3  1      OPC=nop             
  nop                          #  47    0x379d4  1      OPC=nop             
  nop                          #  48    0x379d5  1      OPC=nop             
  nop                          #  49    0x379d6  1      OPC=nop             
  nop                          #  50    0x379d7  1      OPC=nop             
  nop                          #  51    0x379d8  1      OPC=nop             
  nop                          #  52    0x379d9  1      OPC=nop             
  nop                          #  53    0x379da  1      OPC=nop             
  callq .__nacl_read_tp        #  54    0x379db  5      OPC=callq_label     
  leaq -0x480(%rax), %rax      #  55    0x379e0  7      OPC=leaq_r64_m16    
  movl %ebx, %edx              #  56    0x379e7  2      OPC=movl_r32_r32    
  movl $0x10039eee, %esi       #  57    0x379e9  5      OPC=movl_r32_imm32  
  movl %eax, %eax              #  58    0x379ee  2      OPC=movl_r32_r32    
  movl (%r15,%rax,1), %eax     #  59    0x379f0  4      OPC=movl_r32_m32    
  movl %eax, %eax              #  60    0x379f4  2      OPC=movl_r32_r32    
  movl 0xc(%r15,%rax,1), %edi  #  61    0x379f6  5      OPC=movl_r32_m32    
  xorl %eax, %eax              #  62    0x379fb  2      OPC=xorl_r32_r32    
  nop                          #  63    0x379fd  1      OPC=nop             
  nop                          #  64    0x379fe  1      OPC=nop             
  nop                          #  65    0x379ff  1      OPC=nop             
  nop                          #  66    0x37a00  1      OPC=nop             
  nop                          #  67    0x37a01  1      OPC=nop             
  nop                          #  68    0x37a02  1      OPC=nop             
  nop                          #  69    0x37a03  1      OPC=nop             
  nop                          #  70    0x37a04  1      OPC=nop             
  nop                          #  71    0x37a05  1      OPC=nop             
  nop                          #  72    0x37a06  1      OPC=nop             
  nop                          #  73    0x37a07  1      OPC=nop             
  nop                          #  74    0x37a08  1      OPC=nop             
  nop                          #  75    0x37a09  1      OPC=nop             
  nop                          #  76    0x37a0a  1      OPC=nop             
  nop                          #  77    0x37a0b  1      OPC=nop             
  nop                          #  78    0x37a0c  1      OPC=nop             
  nop                          #  79    0x37a0d  1      OPC=nop             
  nop                          #  80    0x37a0e  1      OPC=nop             
  nop                          #  81    0x37a0f  1      OPC=nop             
  nop                          #  82    0x37a10  1      OPC=nop             
  nop                          #  83    0x37a11  1      OPC=nop             
  nop                          #  84    0x37a12  1      OPC=nop             
  nop                          #  85    0x37a13  1      OPC=nop             
  nop                          #  86    0x37a14  1      OPC=nop             
  nop                          #  87    0x37a15  1      OPC=nop             
  nop                          #  88    0x37a16  1      OPC=nop             
  nop                          #  89    0x37a17  1      OPC=nop             
  nop                          #  90    0x37a18  1      OPC=nop             
  nop                          #  91    0x37a19  1      OPC=nop             
  nop                          #  92    0x37a1a  1      OPC=nop             
  callq .fprintf               #  93    0x37a1b  5      OPC=callq_label     
  nop                          #  94    0x37a20  1      OPC=nop             
  nop                          #  95    0x37a21  1      OPC=nop             
  nop                          #  96    0x37a22  1      OPC=nop             
  nop                          #  97    0x37a23  1      OPC=nop             
  nop                          #  98    0x37a24  1      OPC=nop             
  nop                          #  99    0x37a25  1      OPC=nop             
  nop                          #  100   0x37a26  1      OPC=nop             
  nop                          #  101   0x37a27  1      OPC=nop             
  nop                          #  102   0x37a28  1      OPC=nop             
  nop                          #  103   0x37a29  1      OPC=nop             
  nop                          #  104   0x37a2a  1      OPC=nop             
  nop                          #  105   0x37a2b  1      OPC=nop             
  nop                          #  106   0x37a2c  1      OPC=nop             
  nop                          #  107   0x37a2d  1      OPC=nop             
  nop                          #  108   0x37a2e  1      OPC=nop             
  nop                          #  109   0x37a2f  1      OPC=nop             
  nop                          #  110   0x37a30  1      OPC=nop             
  nop                          #  111   0x37a31  1      OPC=nop             
  nop                          #  112   0x37a32  1      OPC=nop             
  nop                          #  113   0x37a33  1      OPC=nop             
  nop                          #  114   0x37a34  1      OPC=nop             
  nop                          #  115   0x37a35  1      OPC=nop             
  nop                          #  116   0x37a36  1      OPC=nop             
  nop                          #  117   0x37a37  1      OPC=nop             
  nop                          #  118   0x37a38  1      OPC=nop             
  nop                          #  119   0x37a39  1      OPC=nop             
  nop                          #  120   0x37a3a  1      OPC=nop             
  callq .abort                 #  121   0x37a3b  5      OPC=callq_label     
                                                                            
.size __gmp_assert_fail, .-__gmp_assert_fail

