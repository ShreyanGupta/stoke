  .text
  .globl _ZNSt10moneypunctIcLb0EED0Ev
  .type _ZNSt10moneypunctIcLb0EED0Ev, @function

#! file-offset 0x11b8c0
#! rip-offset  0xdb8c0
#! capacity    192 bytes

# Text                             #  Line  RIP      Bytes  Opcode              
._ZNSt10moneypunctIcLb0EED0Ev:     #        0xdb8c0  0      OPC=<label>         
  pushq %rbx                       #  1     0xdb8c0  1      OPC=pushq_r64_1     
  movl %edi, %ebx                  #  2     0xdb8c1  2      OPC=movl_r32_r32    
  subl $0x10, %esp                 #  3     0xdb8c3  3      OPC=subl_r32_imm8   
  addq %r15, %rsp                  #  4     0xdb8c6  3      OPC=addq_r64_r64    
  movl %ebx, %ebx                  #  5     0xdb8c9  2      OPC=movl_r32_r32    
  movl 0x8(%r15,%rbx,1), %edi      #  6     0xdb8cb  5      OPC=movl_r32_m32    
  movl %ebx, %ebx                  #  7     0xdb8d0  2      OPC=movl_r32_r32    
  movl $0x1003ae48, (%r15,%rbx,1)  #  8     0xdb8d2  8      OPC=movl_m32_imm32  
  testq %rdi, %rdi                 #  9     0xdb8da  3      OPC=testq_r64_r64   
  nop                              #  10    0xdb8dd  1      OPC=nop             
  nop                              #  11    0xdb8de  1      OPC=nop             
  nop                              #  12    0xdb8df  1      OPC=nop             
  je .L_db900                      #  13    0xdb8e0  2      OPC=je_label        
  movl %edi, %edi                  #  14    0xdb8e2  2      OPC=movl_r32_r32    
  movl (%r15,%rdi,1), %eax         #  15    0xdb8e4  4      OPC=movl_r32_m32    
  movl %eax, %eax                  #  16    0xdb8e8  2      OPC=movl_r32_r32    
  movl 0x4(%r15,%rax,1), %eax      #  17    0xdb8ea  5      OPC=movl_r32_m32    
  nop                              #  18    0xdb8ef  1      OPC=nop             
  nop                              #  19    0xdb8f0  1      OPC=nop             
  nop                              #  20    0xdb8f1  1      OPC=nop             
  nop                              #  21    0xdb8f2  1      OPC=nop             
  nop                              #  22    0xdb8f3  1      OPC=nop             
  nop                              #  23    0xdb8f4  1      OPC=nop             
  nop                              #  24    0xdb8f5  1      OPC=nop             
  nop                              #  25    0xdb8f6  1      OPC=nop             
  nop                              #  26    0xdb8f7  1      OPC=nop             
  andl $0xffffffe0, %eax           #  27    0xdb8f8  6      OPC=andl_r32_imm32  
  nop                              #  28    0xdb8fe  1      OPC=nop             
  nop                              #  29    0xdb8ff  1      OPC=nop             
  nop                              #  30    0xdb900  1      OPC=nop             
  addq %r15, %rax                  #  31    0xdb901  3      OPC=addq_r64_r64    
  callq %rax                       #  32    0xdb904  2      OPC=callq_r64       
.L_db900:                          #        0xdb906  0      OPC=<label>         
  movl %ebx, %edi                  #  33    0xdb906  2      OPC=movl_r32_r32    
  nop                              #  34    0xdb908  1      OPC=nop             
  nop                              #  35    0xdb909  1      OPC=nop             
  nop                              #  36    0xdb90a  1      OPC=nop             
  nop                              #  37    0xdb90b  1      OPC=nop             
  nop                              #  38    0xdb90c  1      OPC=nop             
  nop                              #  39    0xdb90d  1      OPC=nop             
  nop                              #  40    0xdb90e  1      OPC=nop             
  nop                              #  41    0xdb90f  1      OPC=nop             
  nop                              #  42    0xdb910  1      OPC=nop             
  nop                              #  43    0xdb911  1      OPC=nop             
  nop                              #  44    0xdb912  1      OPC=nop             
  nop                              #  45    0xdb913  1      OPC=nop             
  nop                              #  46    0xdb914  1      OPC=nop             
  nop                              #  47    0xdb915  1      OPC=nop             
  nop                              #  48    0xdb916  1      OPC=nop             
  nop                              #  49    0xdb917  1      OPC=nop             
  nop                              #  50    0xdb918  1      OPC=nop             
  nop                              #  51    0xdb919  1      OPC=nop             
  nop                              #  52    0xdb91a  1      OPC=nop             
  nop                              #  53    0xdb91b  1      OPC=nop             
  nop                              #  54    0xdb91c  1      OPC=nop             
  nop                              #  55    0xdb91d  1      OPC=nop             
  nop                              #  56    0xdb91e  1      OPC=nop             
  nop                              #  57    0xdb91f  1      OPC=nop             
  nop                              #  58    0xdb920  1      OPC=nop             
  callq ._ZNSt6locale5facetD2Ev    #  59    0xdb921  5      OPC=callq_label     
  movl %ebx, %edi                  #  60    0xdb926  2      OPC=movl_r32_r32    
  addl $0x10, %esp                 #  61    0xdb928  3      OPC=addl_r32_imm8   
  addq %r15, %rsp                  #  62    0xdb92b  3      OPC=addq_r64_r64    
  popq %rbx                        #  63    0xdb92e  1      OPC=popq_r64_1      
  jmpq ._ZdlPv                     #  64    0xdb92f  5      OPC=jmpq_label_1    
  nop                              #  65    0xdb934  1      OPC=nop             
  nop                              #  66    0xdb935  1      OPC=nop             
  nop                              #  67    0xdb936  1      OPC=nop             
  nop                              #  68    0xdb937  1      OPC=nop             
  nop                              #  69    0xdb938  1      OPC=nop             
  nop                              #  70    0xdb939  1      OPC=nop             
  nop                              #  71    0xdb93a  1      OPC=nop             
  nop                              #  72    0xdb93b  1      OPC=nop             
  nop                              #  73    0xdb93c  1      OPC=nop             
  nop                              #  74    0xdb93d  1      OPC=nop             
  nop                              #  75    0xdb93e  1      OPC=nop             
  nop                              #  76    0xdb93f  1      OPC=nop             
  nop                              #  77    0xdb940  1      OPC=nop             
  nop                              #  78    0xdb941  1      OPC=nop             
  nop                              #  79    0xdb942  1      OPC=nop             
  nop                              #  80    0xdb943  1      OPC=nop             
  nop                              #  81    0xdb944  1      OPC=nop             
  nop                              #  82    0xdb945  1      OPC=nop             
  movl %ebx, %edi                  #  83    0xdb946  2      OPC=movl_r32_r32    
  movl %eax, 0x8(%rsp)             #  84    0xdb948  4      OPC=movl_m32_r32    
  nop                              #  85    0xdb94c  1      OPC=nop             
  nop                              #  86    0xdb94d  1      OPC=nop             
  nop                              #  87    0xdb94e  1      OPC=nop             
  nop                              #  88    0xdb94f  1      OPC=nop             
  nop                              #  89    0xdb950  1      OPC=nop             
  nop                              #  90    0xdb951  1      OPC=nop             
  nop                              #  91    0xdb952  1      OPC=nop             
  nop                              #  92    0xdb953  1      OPC=nop             
  nop                              #  93    0xdb954  1      OPC=nop             
  nop                              #  94    0xdb955  1      OPC=nop             
  nop                              #  95    0xdb956  1      OPC=nop             
  nop                              #  96    0xdb957  1      OPC=nop             
  nop                              #  97    0xdb958  1      OPC=nop             
  nop                              #  98    0xdb959  1      OPC=nop             
  nop                              #  99    0xdb95a  1      OPC=nop             
  nop                              #  100   0xdb95b  1      OPC=nop             
  nop                              #  101   0xdb95c  1      OPC=nop             
  nop                              #  102   0xdb95d  1      OPC=nop             
  nop                              #  103   0xdb95e  1      OPC=nop             
  nop                              #  104   0xdb95f  1      OPC=nop             
  nop                              #  105   0xdb960  1      OPC=nop             
  callq ._ZNSt6locale5facetD2Ev    #  106   0xdb961  5      OPC=callq_label     
  movl 0x8(%rsp), %eax             #  107   0xdb966  4      OPC=movl_r32_m32    
  movl %eax, %edi                  #  108   0xdb96a  2      OPC=movl_r32_r32    
  nop                              #  109   0xdb96c  1      OPC=nop             
  nop                              #  110   0xdb96d  1      OPC=nop             
  nop                              #  111   0xdb96e  1      OPC=nop             
  nop                              #  112   0xdb96f  1      OPC=nop             
  nop                              #  113   0xdb970  1      OPC=nop             
  nop                              #  114   0xdb971  1      OPC=nop             
  nop                              #  115   0xdb972  1      OPC=nop             
  nop                              #  116   0xdb973  1      OPC=nop             
  nop                              #  117   0xdb974  1      OPC=nop             
  nop                              #  118   0xdb975  1      OPC=nop             
  nop                              #  119   0xdb976  1      OPC=nop             
  nop                              #  120   0xdb977  1      OPC=nop             
  nop                              #  121   0xdb978  1      OPC=nop             
  nop                              #  122   0xdb979  1      OPC=nop             
  nop                              #  123   0xdb97a  1      OPC=nop             
  nop                              #  124   0xdb97b  1      OPC=nop             
  nop                              #  125   0xdb97c  1      OPC=nop             
  nop                              #  126   0xdb97d  1      OPC=nop             
  nop                              #  127   0xdb97e  1      OPC=nop             
  nop                              #  128   0xdb97f  1      OPC=nop             
  nop                              #  129   0xdb980  1      OPC=nop             
  callq ._Unwind_Resume            #  130   0xdb981  5      OPC=callq_label     
                                                                                
.size _ZNSt10moneypunctIcLb0EED0Ev, .-_ZNSt10moneypunctIcLb0EED0Ev

