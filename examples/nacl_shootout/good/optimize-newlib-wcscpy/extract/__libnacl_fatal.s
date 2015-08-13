  .text
  .globl __libnacl_fatal
  .type __libnacl_fatal, @function

#! file-offset 0x1582a0
#! rip-offset  0x1182a0
#! capacity    160 bytes

# Text                        #  Line  RIP       Bytes  Opcode              
.__libnacl_fatal:             #        0x1182a0  0      OPC=<label>         
  subl $0x18, %esp            #  1     0x1182a0  3      OPC=subl_r32_imm8   
  addq %r15, %rsp             #  2     0x1182a3  3      OPC=addq_r64_r64    
  movl 0xff586a4(%rip), %eax  #  3     0x1182a6  6      OPC=movl_r32_m32    
  movl %edi, %esi             #  4     0x1182ac  2      OPC=movl_r32_r32    
  testq %rax, %rax            #  5     0x1182ae  3      OPC=testq_r64_r64   
  je .L_118320                #  6     0x1182b1  2      OPC=je_label        
  xorl %edx, %edx             #  7     0x1182b3  2      OPC=xorl_r32_r32    
  movl %esi, %esi             #  8     0x1182b5  2      OPC=movl_r32_r32    
  cmpb $0x0, (%r15,%rsi,1)    #  9     0x1182b7  5      OPC=cmpb_m8_imm8    
  nop                         #  10    0x1182bc  1      OPC=nop             
  nop                         #  11    0x1182bd  1      OPC=nop             
  nop                         #  12    0x1182be  1      OPC=nop             
  nop                         #  13    0x1182bf  1      OPC=nop             
  je .L_118300                #  14    0x1182c0  2      OPC=je_label        
  leal 0x1(%rsi), %ecx        #  15    0x1182c2  3      OPC=leal_r32_m16    
  nop                         #  16    0x1182c5  1      OPC=nop             
  nop                         #  17    0x1182c6  1      OPC=nop             
  nop                         #  18    0x1182c7  1      OPC=nop             
  nop                         #  19    0x1182c8  1      OPC=nop             
  nop                         #  20    0x1182c9  1      OPC=nop             
  nop                         #  21    0x1182ca  1      OPC=nop             
  nop                         #  22    0x1182cb  1      OPC=nop             
  nop                         #  23    0x1182cc  1      OPC=nop             
  nop                         #  24    0x1182cd  1      OPC=nop             
  nop                         #  25    0x1182ce  1      OPC=nop             
  nop                         #  26    0x1182cf  1      OPC=nop             
  nop                         #  27    0x1182d0  1      OPC=nop             
  nop                         #  28    0x1182d1  1      OPC=nop             
  nop                         #  29    0x1182d2  1      OPC=nop             
  nop                         #  30    0x1182d3  1      OPC=nop             
  nop                         #  31    0x1182d4  1      OPC=nop             
  nop                         #  32    0x1182d5  1      OPC=nop             
  nop                         #  33    0x1182d6  1      OPC=nop             
  nop                         #  34    0x1182d7  1      OPC=nop             
  nop                         #  35    0x1182d8  1      OPC=nop             
  nop                         #  36    0x1182d9  1      OPC=nop             
  nop                         #  37    0x1182da  1      OPC=nop             
  nop                         #  38    0x1182db  1      OPC=nop             
  nop                         #  39    0x1182dc  1      OPC=nop             
  nop                         #  40    0x1182dd  1      OPC=nop             
  nop                         #  41    0x1182de  1      OPC=nop             
  nop                         #  42    0x1182df  1      OPC=nop             
.L_1182e0:                    #        0x1182e0  0      OPC=<label>         
  movl %ecx, %ecx             #  43    0x1182e0  2      OPC=movl_r32_r32    
  movzbl (%r15,%rcx,1), %edi  #  44    0x1182e2  5      OPC=movzbl_r32_m8   
  addl $0x1, %edx             #  45    0x1182e7  3      OPC=addl_r32_imm8   
  addl $0x1, %ecx             #  46    0x1182ea  3      OPC=addl_r32_imm8   
  testb %dil, %dil            #  47    0x1182ed  3      OPC=testb_r8_r8     
  jne .L_1182e0               #  48    0x1182f0  2      OPC=jne_label       
  nop                         #  49    0x1182f2  1      OPC=nop             
  nop                         #  50    0x1182f3  1      OPC=nop             
  nop                         #  51    0x1182f4  1      OPC=nop             
  nop                         #  52    0x1182f5  1      OPC=nop             
  nop                         #  53    0x1182f6  1      OPC=nop             
  nop                         #  54    0x1182f7  1      OPC=nop             
  nop                         #  55    0x1182f8  1      OPC=nop             
  nop                         #  56    0x1182f9  1      OPC=nop             
  nop                         #  57    0x1182fa  1      OPC=nop             
  nop                         #  58    0x1182fb  1      OPC=nop             
  nop                         #  59    0x1182fc  1      OPC=nop             
  nop                         #  60    0x1182fd  1      OPC=nop             
  nop                         #  61    0x1182fe  1      OPC=nop             
  nop                         #  62    0x1182ff  1      OPC=nop             
.L_118300:                    #        0x118300  0      OPC=<label>         
  leal 0xc(%rsp), %ecx        #  63    0x118300  4      OPC=leal_r32_m16    
  movl $0x2, %edi             #  64    0x118304  5      OPC=movl_r32_imm32  
  nop                         #  65    0x118309  1      OPC=nop             
  nop                         #  66    0x11830a  1      OPC=nop             
  nop                         #  67    0x11830b  1      OPC=nop             
  nop                         #  68    0x11830c  1      OPC=nop             
  nop                         #  69    0x11830d  1      OPC=nop             
  nop                         #  70    0x11830e  1      OPC=nop             
  nop                         #  71    0x11830f  1      OPC=nop             
  nop                         #  72    0x118310  1      OPC=nop             
  nop                         #  73    0x118311  1      OPC=nop             
  nop                         #  74    0x118312  1      OPC=nop             
  nop                         #  75    0x118313  1      OPC=nop             
  nop                         #  76    0x118314  1      OPC=nop             
  nop                         #  77    0x118315  1      OPC=nop             
  nop                         #  78    0x118316  1      OPC=nop             
  nop                         #  79    0x118317  1      OPC=nop             
  andl $0xffffffe0, %eax      #  80    0x118318  6      OPC=andl_r32_imm32  
  nop                         #  81    0x11831e  1      OPC=nop             
  nop                         #  82    0x11831f  1      OPC=nop             
  nop                         #  83    0x118320  1      OPC=nop             
  addq %r15, %rax             #  84    0x118321  3      OPC=addq_r64_r64    
  callq %rax                  #  85    0x118324  2      OPC=callq_r64       
.L_118320:                    #        0x118326  0      OPC=<label>         
  ud2                         #  86    0x118326  2      OPC=ud2             
  nop                         #  87    0x118328  1      OPC=nop             
  nop                         #  88    0x118329  1      OPC=nop             
  nop                         #  89    0x11832a  1      OPC=nop             
  nop                         #  90    0x11832b  1      OPC=nop             
  nop                         #  91    0x11832c  1      OPC=nop             
  nop                         #  92    0x11832d  1      OPC=nop             
  nop                         #  93    0x11832e  1      OPC=nop             
  nop                         #  94    0x11832f  1      OPC=nop             
  nop                         #  95    0x118330  1      OPC=nop             
  nop                         #  96    0x118331  1      OPC=nop             
  nop                         #  97    0x118332  1      OPC=nop             
  nop                         #  98    0x118333  1      OPC=nop             
  nop                         #  99    0x118334  1      OPC=nop             
  nop                         #  100   0x118335  1      OPC=nop             
  nop                         #  101   0x118336  1      OPC=nop             
  nop                         #  102   0x118337  1      OPC=nop             
  nop                         #  103   0x118338  1      OPC=nop             
  nop                         #  104   0x118339  1      OPC=nop             
  nop                         #  105   0x11833a  1      OPC=nop             
  nop                         #  106   0x11833b  1      OPC=nop             
  nop                         #  107   0x11833c  1      OPC=nop             
  nop                         #  108   0x11833d  1      OPC=nop             
  nop                         #  109   0x11833e  1      OPC=nop             
  nop                         #  110   0x11833f  1      OPC=nop             
  nop                         #  111   0x118340  1      OPC=nop             
  nop                         #  112   0x118341  1      OPC=nop             
  nop                         #  113   0x118342  1      OPC=nop             
  nop                         #  114   0x118343  1      OPC=nop             
  nop                         #  115   0x118344  1      OPC=nop             
  nop                         #  116   0x118345  1      OPC=nop             
                                                                            
.size __libnacl_fatal, .-__libnacl_fatal

