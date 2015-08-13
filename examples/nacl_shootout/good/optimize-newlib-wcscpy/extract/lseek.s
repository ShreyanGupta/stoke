  .text
  .globl lseek
  .type lseek, @function

#! file-offset 0x158680
#! rip-offset  0x118680
#! capacity    160 bytes

# Text                        #  Line  RIP       Bytes  Opcode              
.lseek:                       #        0x118680  0      OPC=<label>         
  pushq %rbx                  #  1     0x118680  1      OPC=pushq_r64_1     
  subl $0x10, %esp            #  2     0x118681  3      OPC=subl_r32_imm8   
  addq %r15, %rsp             #  3     0x118684  3      OPC=addq_r64_r64    
  movl 0xff582c7(%rip), %eax  #  4     0x118687  6      OPC=movl_r32_m32    
  leal 0x8(%rsp), %ecx        #  5     0x11868d  4      OPC=leal_r32_m16    
  nop                         #  6     0x118691  1      OPC=nop             
  nop                         #  7     0x118692  1      OPC=nop             
  nop                         #  8     0x118693  1      OPC=nop             
  nop                         #  9     0x118694  1      OPC=nop             
  nop                         #  10    0x118695  1      OPC=nop             
  nop                         #  11    0x118696  1      OPC=nop             
  nop                         #  12    0x118697  1      OPC=nop             
  andl $0xffffffe0, %eax      #  13    0x118698  6      OPC=andl_r32_imm32  
  nop                         #  14    0x11869e  1      OPC=nop             
  nop                         #  15    0x11869f  1      OPC=nop             
  nop                         #  16    0x1186a0  1      OPC=nop             
  addq %r15, %rax             #  17    0x1186a1  3      OPC=addq_r64_r64    
  callq %rax                  #  18    0x1186a4  2      OPC=callq_r64       
  testl %eax, %eax            #  19    0x1186a6  2      OPC=testl_r32_r32   
  movl %eax, %ebx             #  20    0x1186a8  2      OPC=movl_r32_r32    
  jne .L_1186e0               #  21    0x1186aa  2      OPC=jne_label       
  movq 0x8(%rsp), %rax        #  22    0x1186ac  5      OPC=movq_r64_m64    
  nop                         #  23    0x1186b1  1      OPC=nop             
  nop                         #  24    0x1186b2  1      OPC=nop             
  nop                         #  25    0x1186b3  1      OPC=nop             
  nop                         #  26    0x1186b4  1      OPC=nop             
  nop                         #  27    0x1186b5  1      OPC=nop             
  nop                         #  28    0x1186b6  1      OPC=nop             
  nop                         #  29    0x1186b7  1      OPC=nop             
  nop                         #  30    0x1186b8  1      OPC=nop             
  nop                         #  31    0x1186b9  1      OPC=nop             
  nop                         #  32    0x1186ba  1      OPC=nop             
  nop                         #  33    0x1186bb  1      OPC=nop             
  nop                         #  34    0x1186bc  1      OPC=nop             
  nop                         #  35    0x1186bd  1      OPC=nop             
  nop                         #  36    0x1186be  1      OPC=nop             
  nop                         #  37    0x1186bf  1      OPC=nop             
  nop                         #  38    0x1186c0  1      OPC=nop             
  nop                         #  39    0x1186c1  1      OPC=nop             
  nop                         #  40    0x1186c2  1      OPC=nop             
  nop                         #  41    0x1186c3  1      OPC=nop             
  nop                         #  42    0x1186c4  1      OPC=nop             
  nop                         #  43    0x1186c5  1      OPC=nop             
.L_1186c0:                    #        0x1186c6  0      OPC=<label>         
  addl $0x10, %esp            #  44    0x1186c6  3      OPC=addl_r32_imm8   
  addq %r15, %rsp             #  45    0x1186c9  3      OPC=addq_r64_r64    
  popq %rbx                   #  46    0x1186cc  1      OPC=popq_r64_1      
  popq %r11                   #  47    0x1186cd  2      OPC=popq_r64_1      
  andl $0xffffffe0, %r11d     #  48    0x1186cf  7      OPC=andl_r32_imm32  
  nop                         #  49    0x1186d6  1      OPC=nop             
  nop                         #  50    0x1186d7  1      OPC=nop             
  nop                         #  51    0x1186d8  1      OPC=nop             
  nop                         #  52    0x1186d9  1      OPC=nop             
  addq %r15, %r11             #  53    0x1186da  3      OPC=addq_r64_r64    
  jmpq %r11                   #  54    0x1186dd  3      OPC=jmpq_r64        
  nop                         #  55    0x1186e0  1      OPC=nop             
  nop                         #  56    0x1186e1  1      OPC=nop             
  nop                         #  57    0x1186e2  1      OPC=nop             
  nop                         #  58    0x1186e3  1      OPC=nop             
  nop                         #  59    0x1186e4  1      OPC=nop             
  nop                         #  60    0x1186e5  1      OPC=nop             
  nop                         #  61    0x1186e6  1      OPC=nop             
  nop                         #  62    0x1186e7  1      OPC=nop             
  nop                         #  63    0x1186e8  1      OPC=nop             
  nop                         #  64    0x1186e9  1      OPC=nop             
  nop                         #  65    0x1186ea  1      OPC=nop             
  nop                         #  66    0x1186eb  1      OPC=nop             
  nop                         #  67    0x1186ec  1      OPC=nop             
.L_1186e0:                    #        0x1186ed  0      OPC=<label>         
  nop                         #  68    0x1186ed  1      OPC=nop             
  nop                         #  69    0x1186ee  1      OPC=nop             
  nop                         #  70    0x1186ef  1      OPC=nop             
  nop                         #  71    0x1186f0  1      OPC=nop             
  nop                         #  72    0x1186f1  1      OPC=nop             
  nop                         #  73    0x1186f2  1      OPC=nop             
  nop                         #  74    0x1186f3  1      OPC=nop             
  nop                         #  75    0x1186f4  1      OPC=nop             
  nop                         #  76    0x1186f5  1      OPC=nop             
  nop                         #  77    0x1186f6  1      OPC=nop             
  nop                         #  78    0x1186f7  1      OPC=nop             
  nop                         #  79    0x1186f8  1      OPC=nop             
  nop                         #  80    0x1186f9  1      OPC=nop             
  nop                         #  81    0x1186fa  1      OPC=nop             
  nop                         #  82    0x1186fb  1      OPC=nop             
  nop                         #  83    0x1186fc  1      OPC=nop             
  nop                         #  84    0x1186fd  1      OPC=nop             
  nop                         #  85    0x1186fe  1      OPC=nop             
  nop                         #  86    0x1186ff  1      OPC=nop             
  nop                         #  87    0x118700  1      OPC=nop             
  nop                         #  88    0x118701  1      OPC=nop             
  nop                         #  89    0x118702  1      OPC=nop             
  nop                         #  90    0x118703  1      OPC=nop             
  nop                         #  91    0x118704  1      OPC=nop             
  nop                         #  92    0x118705  1      OPC=nop             
  nop                         #  93    0x118706  1      OPC=nop             
  nop                         #  94    0x118707  1      OPC=nop             
  callq .__errno              #  95    0x118708  5      OPC=callq_label     
  movl %eax, %eax             #  96    0x11870d  2      OPC=movl_r32_r32    
  movl %eax, %eax             #  97    0x11870f  2      OPC=movl_r32_r32    
  movl %ebx, (%r15,%rax,1)    #  98    0x118711  4      OPC=movl_m32_r32    
  movq $0xffffffff, %rax      #  99    0x118715  7      OPC=movq_r64_imm32  
  jmpq .L_1186c0              #  100   0x11871c  2      OPC=jmpq_label      
  nop                         #  101   0x11871e  1      OPC=nop             
  nop                         #  102   0x11871f  1      OPC=nop             
  nop                         #  103   0x118720  1      OPC=nop             
  nop                         #  104   0x118721  1      OPC=nop             
  nop                         #  105   0x118722  1      OPC=nop             
  nop                         #  106   0x118723  1      OPC=nop             
  nop                         #  107   0x118724  1      OPC=nop             
  nop                         #  108   0x118725  1      OPC=nop             
  nop                         #  109   0x118726  1      OPC=nop             
  nop                         #  110   0x118727  1      OPC=nop             
  nop                         #  111   0x118728  1      OPC=nop             
  nop                         #  112   0x118729  1      OPC=nop             
  nop                         #  113   0x11872a  1      OPC=nop             
  nop                         #  114   0x11872b  1      OPC=nop             
  nop                         #  115   0x11872c  1      OPC=nop             
                                                                            
.size lseek, .-lseek

