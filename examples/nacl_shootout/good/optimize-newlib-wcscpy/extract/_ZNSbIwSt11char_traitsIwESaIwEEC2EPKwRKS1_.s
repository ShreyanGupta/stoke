  .text
  .globl _ZNSbIwSt11char_traitsIwESaIwEEC2EPKwRKS1_
  .type _ZNSbIwSt11char_traitsIwESaIwEEC2EPKwRKS1_, @function

#! file-offset 0x117260
#! rip-offset  0xd7260
#! capacity    192 bytes

# Text                                                                                            #  Line  RIP      Bytes  Opcode                
._ZNSbIwSt11char_traitsIwESaIwEEC2EPKwRKS1_:                                                      #        0xd7260  0      OPC=<label>           
  movq %rbx, -0x18(%rsp)                                                                          #  1     0xd7260  5      OPC=movq_m64_r64      
  movl %esi, %ebx                                                                                 #  2     0xd7265  2      OPC=movl_r32_r32      
  movq %r12, -0x10(%rsp)                                                                          #  3     0xd7267  5      OPC=movq_m64_r64      
  movq %r13, -0x8(%rsp)                                                                           #  4     0xd726c  5      OPC=movq_m64_r64      
  subl $0x28, %esp                                                                                #  5     0xd7271  3      OPC=subl_r32_imm8     
  addq %r15, %rsp                                                                                 #  6     0xd7274  3      OPC=addq_r64_r64      
  testq %rbx, %rbx                                                                                #  7     0xd7277  3      OPC=testq_r64_r64     
  movl %edi, %r12d                                                                                #  8     0xd727a  3      OPC=movl_r32_r32      
  movl %edx, %r13d                                                                                #  9     0xd727d  3      OPC=movl_r32_r32      
  movl $0xfffffffc, %esi                                                                          #  10    0xd7280  6      OPC=movl_r32_imm32_1  
  je .L_d72c0                                                                                     #  11    0xd7286  2      OPC=je_label          
  movl %ebx, %edi                                                                                 #  12    0xd7288  2      OPC=movl_r32_r32      
  nop                                                                                             #  13    0xd728a  1      OPC=nop               
  nop                                                                                             #  14    0xd728b  1      OPC=nop               
  nop                                                                                             #  15    0xd728c  1      OPC=nop               
  nop                                                                                             #  16    0xd728d  1      OPC=nop               
  nop                                                                                             #  17    0xd728e  1      OPC=nop               
  nop                                                                                             #  18    0xd728f  1      OPC=nop               
  nop                                                                                             #  19    0xd7290  1      OPC=nop               
  nop                                                                                             #  20    0xd7291  1      OPC=nop               
  nop                                                                                             #  21    0xd7292  1      OPC=nop               
  nop                                                                                             #  22    0xd7293  1      OPC=nop               
  nop                                                                                             #  23    0xd7294  1      OPC=nop               
  nop                                                                                             #  24    0xd7295  1      OPC=nop               
  nop                                                                                             #  25    0xd7296  1      OPC=nop               
  nop                                                                                             #  26    0xd7297  1      OPC=nop               
  nop                                                                                             #  27    0xd7298  1      OPC=nop               
  nop                                                                                             #  28    0xd7299  1      OPC=nop               
  nop                                                                                             #  29    0xd729a  1      OPC=nop               
  nop                                                                                             #  30    0xd729b  1      OPC=nop               
  callq .wcslen                                                                                   #  31    0xd729c  5      OPC=callq_label       
  leal (%rbx,%rax,4), %esi                                                                        #  32    0xd72a1  3      OPC=leal_r32_m16      
  nop                                                                                             #  33    0xd72a4  1      OPC=nop               
  nop                                                                                             #  34    0xd72a5  1      OPC=nop               
  nop                                                                                             #  35    0xd72a6  1      OPC=nop               
  nop                                                                                             #  36    0xd72a7  1      OPC=nop               
  nop                                                                                             #  37    0xd72a8  1      OPC=nop               
  nop                                                                                             #  38    0xd72a9  1      OPC=nop               
  nop                                                                                             #  39    0xd72aa  1      OPC=nop               
  nop                                                                                             #  40    0xd72ab  1      OPC=nop               
  nop                                                                                             #  41    0xd72ac  1      OPC=nop               
  nop                                                                                             #  42    0xd72ad  1      OPC=nop               
  nop                                                                                             #  43    0xd72ae  1      OPC=nop               
  nop                                                                                             #  44    0xd72af  1      OPC=nop               
  nop                                                                                             #  45    0xd72b0  1      OPC=nop               
  nop                                                                                             #  46    0xd72b1  1      OPC=nop               
  nop                                                                                             #  47    0xd72b2  1      OPC=nop               
  nop                                                                                             #  48    0xd72b3  1      OPC=nop               
  nop                                                                                             #  49    0xd72b4  1      OPC=nop               
  nop                                                                                             #  50    0xd72b5  1      OPC=nop               
  nop                                                                                             #  51    0xd72b6  1      OPC=nop               
  nop                                                                                             #  52    0xd72b7  1      OPC=nop               
  nop                                                                                             #  53    0xd72b8  1      OPC=nop               
  nop                                                                                             #  54    0xd72b9  1      OPC=nop               
  nop                                                                                             #  55    0xd72ba  1      OPC=nop               
  nop                                                                                             #  56    0xd72bb  1      OPC=nop               
  nop                                                                                             #  57    0xd72bc  1      OPC=nop               
  nop                                                                                             #  58    0xd72bd  1      OPC=nop               
  nop                                                                                             #  59    0xd72be  1      OPC=nop               
  nop                                                                                             #  60    0xd72bf  1      OPC=nop               
  nop                                                                                             #  61    0xd72c0  1      OPC=nop               
.L_d72c0:                                                                                         #        0xd72c1  0      OPC=<label>           
  movl %r13d, %edx                                                                                #  62    0xd72c1  3      OPC=movl_r32_r32      
  movl %ebx, %edi                                                                                 #  63    0xd72c4  2      OPC=movl_r32_r32      
  movb $0x0, (%rsp)                                                                               #  64    0xd72c6  4      OPC=movb_m8_imm8      
  nop                                                                                             #  65    0xd72ca  1      OPC=nop               
  nop                                                                                             #  66    0xd72cb  1      OPC=nop               
  nop                                                                                             #  67    0xd72cc  1      OPC=nop               
  nop                                                                                             #  68    0xd72cd  1      OPC=nop               
  nop                                                                                             #  69    0xd72ce  1      OPC=nop               
  nop                                                                                             #  70    0xd72cf  1      OPC=nop               
  nop                                                                                             #  71    0xd72d0  1      OPC=nop               
  nop                                                                                             #  72    0xd72d1  1      OPC=nop               
  nop                                                                                             #  73    0xd72d2  1      OPC=nop               
  nop                                                                                             #  74    0xd72d3  1      OPC=nop               
  nop                                                                                             #  75    0xd72d4  1      OPC=nop               
  nop                                                                                             #  76    0xd72d5  1      OPC=nop               
  nop                                                                                             #  77    0xd72d6  1      OPC=nop               
  nop                                                                                             #  78    0xd72d7  1      OPC=nop               
  nop                                                                                             #  79    0xd72d8  1      OPC=nop               
  nop                                                                                             #  80    0xd72d9  1      OPC=nop               
  nop                                                                                             #  81    0xd72da  1      OPC=nop               
  nop                                                                                             #  82    0xd72db  1      OPC=nop               
  callq ._ZNSbIwSt11char_traitsIwESaIwEE12_S_constructIPKwEEPwT_S7_RKS1_St20forward_iterator_tag  #  83    0xd72dc  5      OPC=callq_label       
  movq 0x10(%rsp), %rbx                                                                           #  84    0xd72e1  5      OPC=movq_r64_m64      
  movl %r12d, %r12d                                                                               #  85    0xd72e6  3      OPC=movl_r32_r32      
  movl %eax, (%r15,%r12,1)                                                                        #  86    0xd72e9  4      OPC=movl_m32_r32      
  movq 0x20(%rsp), %r13                                                                           #  87    0xd72ed  5      OPC=movq_r64_m64      
  movq 0x18(%rsp), %r12                                                                           #  88    0xd72f2  5      OPC=movq_r64_m64      
  addl $0x28, %esp                                                                                #  89    0xd72f7  3      OPC=addl_r32_imm8     
  addq %r15, %rsp                                                                                 #  90    0xd72fa  3      OPC=addq_r64_r64      
  popq %r11                                                                                       #  91    0xd72fd  2      OPC=popq_r64_1        
  xchgw %ax, %ax                                                                                  #  92    0xd72ff  2      OPC=xchgw_ax_r16      
  andl $0xffffffe0, %r11d                                                                         #  93    0xd7301  7      OPC=andl_r32_imm32    
  nop                                                                                             #  94    0xd7308  1      OPC=nop               
  nop                                                                                             #  95    0xd7309  1      OPC=nop               
  nop                                                                                             #  96    0xd730a  1      OPC=nop               
  nop                                                                                             #  97    0xd730b  1      OPC=nop               
  addq %r15, %r11                                                                                 #  98    0xd730c  3      OPC=addq_r64_r64      
  jmpq %r11                                                                                       #  99    0xd730f  3      OPC=jmpq_r64          
  nop                                                                                             #  100   0xd7312  1      OPC=nop               
  nop                                                                                             #  101   0xd7313  1      OPC=nop               
  nop                                                                                             #  102   0xd7314  1      OPC=nop               
  nop                                                                                             #  103   0xd7315  1      OPC=nop               
  nop                                                                                             #  104   0xd7316  1      OPC=nop               
  nop                                                                                             #  105   0xd7317  1      OPC=nop               
  nop                                                                                             #  106   0xd7318  1      OPC=nop               
  nop                                                                                             #  107   0xd7319  1      OPC=nop               
  nop                                                                                             #  108   0xd731a  1      OPC=nop               
  nop                                                                                             #  109   0xd731b  1      OPC=nop               
  nop                                                                                             #  110   0xd731c  1      OPC=nop               
  nop                                                                                             #  111   0xd731d  1      OPC=nop               
  nop                                                                                             #  112   0xd731e  1      OPC=nop               
  nop                                                                                             #  113   0xd731f  1      OPC=nop               
  nop                                                                                             #  114   0xd7320  1      OPC=nop               
  nop                                                                                             #  115   0xd7321  1      OPC=nop               
  nop                                                                                             #  116   0xd7322  1      OPC=nop               
  nop                                                                                             #  117   0xd7323  1      OPC=nop               
  nop                                                                                             #  118   0xd7324  1      OPC=nop               
  nop                                                                                             #  119   0xd7325  1      OPC=nop               
  nop                                                                                             #  120   0xd7326  1      OPC=nop               
  nop                                                                                             #  121   0xd7327  1      OPC=nop               
                                                                                                                                                 
.size _ZNSbIwSt11char_traitsIwESaIwEEC2EPKwRKS1_, .-_ZNSbIwSt11char_traitsIwESaIwEEC2EPKwRKS1_

