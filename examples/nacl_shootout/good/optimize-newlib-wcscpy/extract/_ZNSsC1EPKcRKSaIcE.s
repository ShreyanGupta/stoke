  .text
  .globl _ZNSsC1EPKcRKSaIcE
  .type _ZNSsC1EPKcRKSaIcE, @function

#! file-offset 0xed460
#! rip-offset  0xad460
#! capacity    192 bytes

# Text                                                                    #  Line  RIP      Bytes  Opcode                
._ZNSsC1EPKcRKSaIcE:                                                      #        0xad460  0      OPC=<label>           
  movq %rbx, -0x18(%rsp)                                                  #  1     0xad460  5      OPC=movq_m64_r64      
  movl %esi, %ebx                                                         #  2     0xad465  2      OPC=movl_r32_r32      
  movq %r12, -0x10(%rsp)                                                  #  3     0xad467  5      OPC=movq_m64_r64      
  movq %r13, -0x8(%rsp)                                                   #  4     0xad46c  5      OPC=movq_m64_r64      
  subl $0x28, %esp                                                        #  5     0xad471  3      OPC=subl_r32_imm8     
  addq %r15, %rsp                                                         #  6     0xad474  3      OPC=addq_r64_r64      
  testq %rbx, %rbx                                                        #  7     0xad477  3      OPC=testq_r64_r64     
  movl %edi, %r12d                                                        #  8     0xad47a  3      OPC=movl_r32_r32      
  movl %edx, %r13d                                                        #  9     0xad47d  3      OPC=movl_r32_r32      
  movl $0xffffffff, %esi                                                  #  10    0xad480  6      OPC=movl_r32_imm32_1  
  je .L_ad4c0                                                             #  11    0xad486  2      OPC=je_label          
  movl %ebx, %edi                                                         #  12    0xad488  2      OPC=movl_r32_r32      
  nop                                                                     #  13    0xad48a  1      OPC=nop               
  nop                                                                     #  14    0xad48b  1      OPC=nop               
  nop                                                                     #  15    0xad48c  1      OPC=nop               
  nop                                                                     #  16    0xad48d  1      OPC=nop               
  nop                                                                     #  17    0xad48e  1      OPC=nop               
  nop                                                                     #  18    0xad48f  1      OPC=nop               
  nop                                                                     #  19    0xad490  1      OPC=nop               
  nop                                                                     #  20    0xad491  1      OPC=nop               
  nop                                                                     #  21    0xad492  1      OPC=nop               
  nop                                                                     #  22    0xad493  1      OPC=nop               
  nop                                                                     #  23    0xad494  1      OPC=nop               
  nop                                                                     #  24    0xad495  1      OPC=nop               
  nop                                                                     #  25    0xad496  1      OPC=nop               
  nop                                                                     #  26    0xad497  1      OPC=nop               
  nop                                                                     #  27    0xad498  1      OPC=nop               
  nop                                                                     #  28    0xad499  1      OPC=nop               
  nop                                                                     #  29    0xad49a  1      OPC=nop               
  nop                                                                     #  30    0xad49b  1      OPC=nop               
  callq .strlen                                                           #  31    0xad49c  5      OPC=callq_label       
  leal (%rax,%rbx,1), %esi                                                #  32    0xad4a1  3      OPC=leal_r32_m16      
  nop                                                                     #  33    0xad4a4  1      OPC=nop               
  nop                                                                     #  34    0xad4a5  1      OPC=nop               
  nop                                                                     #  35    0xad4a6  1      OPC=nop               
  nop                                                                     #  36    0xad4a7  1      OPC=nop               
  nop                                                                     #  37    0xad4a8  1      OPC=nop               
  nop                                                                     #  38    0xad4a9  1      OPC=nop               
  nop                                                                     #  39    0xad4aa  1      OPC=nop               
  nop                                                                     #  40    0xad4ab  1      OPC=nop               
  nop                                                                     #  41    0xad4ac  1      OPC=nop               
  nop                                                                     #  42    0xad4ad  1      OPC=nop               
  nop                                                                     #  43    0xad4ae  1      OPC=nop               
  nop                                                                     #  44    0xad4af  1      OPC=nop               
  nop                                                                     #  45    0xad4b0  1      OPC=nop               
  nop                                                                     #  46    0xad4b1  1      OPC=nop               
  nop                                                                     #  47    0xad4b2  1      OPC=nop               
  nop                                                                     #  48    0xad4b3  1      OPC=nop               
  nop                                                                     #  49    0xad4b4  1      OPC=nop               
  nop                                                                     #  50    0xad4b5  1      OPC=nop               
  nop                                                                     #  51    0xad4b6  1      OPC=nop               
  nop                                                                     #  52    0xad4b7  1      OPC=nop               
  nop                                                                     #  53    0xad4b8  1      OPC=nop               
  nop                                                                     #  54    0xad4b9  1      OPC=nop               
  nop                                                                     #  55    0xad4ba  1      OPC=nop               
  nop                                                                     #  56    0xad4bb  1      OPC=nop               
  nop                                                                     #  57    0xad4bc  1      OPC=nop               
  nop                                                                     #  58    0xad4bd  1      OPC=nop               
  nop                                                                     #  59    0xad4be  1      OPC=nop               
  nop                                                                     #  60    0xad4bf  1      OPC=nop               
  nop                                                                     #  61    0xad4c0  1      OPC=nop               
.L_ad4c0:                                                                 #        0xad4c1  0      OPC=<label>           
  movl %r13d, %edx                                                        #  62    0xad4c1  3      OPC=movl_r32_r32      
  movl %ebx, %edi                                                         #  63    0xad4c4  2      OPC=movl_r32_r32      
  movb $0x0, (%rsp)                                                       #  64    0xad4c6  4      OPC=movb_m8_imm8      
  nop                                                                     #  65    0xad4ca  1      OPC=nop               
  nop                                                                     #  66    0xad4cb  1      OPC=nop               
  nop                                                                     #  67    0xad4cc  1      OPC=nop               
  nop                                                                     #  68    0xad4cd  1      OPC=nop               
  nop                                                                     #  69    0xad4ce  1      OPC=nop               
  nop                                                                     #  70    0xad4cf  1      OPC=nop               
  nop                                                                     #  71    0xad4d0  1      OPC=nop               
  nop                                                                     #  72    0xad4d1  1      OPC=nop               
  nop                                                                     #  73    0xad4d2  1      OPC=nop               
  nop                                                                     #  74    0xad4d3  1      OPC=nop               
  nop                                                                     #  75    0xad4d4  1      OPC=nop               
  nop                                                                     #  76    0xad4d5  1      OPC=nop               
  nop                                                                     #  77    0xad4d6  1      OPC=nop               
  nop                                                                     #  78    0xad4d7  1      OPC=nop               
  nop                                                                     #  79    0xad4d8  1      OPC=nop               
  nop                                                                     #  80    0xad4d9  1      OPC=nop               
  nop                                                                     #  81    0xad4da  1      OPC=nop               
  nop                                                                     #  82    0xad4db  1      OPC=nop               
  callq ._ZNSs12_S_constructIPKcEEPcT_S3_RKSaIcESt20forward_iterator_tag  #  83    0xad4dc  5      OPC=callq_label       
  movq 0x10(%rsp), %rbx                                                   #  84    0xad4e1  5      OPC=movq_r64_m64      
  movl %r12d, %r12d                                                       #  85    0xad4e6  3      OPC=movl_r32_r32      
  movl %eax, (%r15,%r12,1)                                                #  86    0xad4e9  4      OPC=movl_m32_r32      
  movq 0x20(%rsp), %r13                                                   #  87    0xad4ed  5      OPC=movq_r64_m64      
  movq 0x18(%rsp), %r12                                                   #  88    0xad4f2  5      OPC=movq_r64_m64      
  addl $0x28, %esp                                                        #  89    0xad4f7  3      OPC=addl_r32_imm8     
  addq %r15, %rsp                                                         #  90    0xad4fa  3      OPC=addq_r64_r64      
  popq %r11                                                               #  91    0xad4fd  2      OPC=popq_r64_1        
  xchgw %ax, %ax                                                          #  92    0xad4ff  2      OPC=xchgw_ax_r16      
  andl $0xffffffe0, %r11d                                                 #  93    0xad501  7      OPC=andl_r32_imm32    
  nop                                                                     #  94    0xad508  1      OPC=nop               
  nop                                                                     #  95    0xad509  1      OPC=nop               
  nop                                                                     #  96    0xad50a  1      OPC=nop               
  nop                                                                     #  97    0xad50b  1      OPC=nop               
  addq %r15, %r11                                                         #  98    0xad50c  3      OPC=addq_r64_r64      
  jmpq %r11                                                               #  99    0xad50f  3      OPC=jmpq_r64          
  nop                                                                     #  100   0xad512  1      OPC=nop               
  nop                                                                     #  101   0xad513  1      OPC=nop               
  nop                                                                     #  102   0xad514  1      OPC=nop               
  nop                                                                     #  103   0xad515  1      OPC=nop               
  nop                                                                     #  104   0xad516  1      OPC=nop               
  nop                                                                     #  105   0xad517  1      OPC=nop               
  nop                                                                     #  106   0xad518  1      OPC=nop               
  nop                                                                     #  107   0xad519  1      OPC=nop               
  nop                                                                     #  108   0xad51a  1      OPC=nop               
  nop                                                                     #  109   0xad51b  1      OPC=nop               
  nop                                                                     #  110   0xad51c  1      OPC=nop               
  nop                                                                     #  111   0xad51d  1      OPC=nop               
  nop                                                                     #  112   0xad51e  1      OPC=nop               
  nop                                                                     #  113   0xad51f  1      OPC=nop               
  nop                                                                     #  114   0xad520  1      OPC=nop               
  nop                                                                     #  115   0xad521  1      OPC=nop               
  nop                                                                     #  116   0xad522  1      OPC=nop               
  nop                                                                     #  117   0xad523  1      OPC=nop               
  nop                                                                     #  118   0xad524  1      OPC=nop               
  nop                                                                     #  119   0xad525  1      OPC=nop               
  nop                                                                     #  120   0xad526  1      OPC=nop               
  nop                                                                     #  121   0xad527  1      OPC=nop               
                                                                                                                         
.size _ZNSsC1EPKcRKSaIcE, .-_ZNSsC1EPKcRKSaIcE

