import React from 'react'

export default function Card({productId , productName ,category ,price, brand,love_count , rating, size,product_id}) {
  return (
    <div className='border border-neutral-800 w-[250px] h-[300px] rounded-2xl relative cursor-pointer hover:scale-105  duration-300'>
      <div className='bg-rose-800 w-full h-[100px] rounded-t-2xl text-xl text-rose-300 text-center pt-6 '>{brand}</div>
      <div className='px-3 pt-1'>
        <div className='flex justify-between items-center'>
            <div className='text-[11px] text-stone-500 font-semibold'>{category}</div>
            <div className='text-[13px] text-orange-500 font-semibold'>{rating ? Number(rating).toFixed(1) : "0.0"}  ⭐</div>
        </div>
        <div className='font-semibold  h-[50px]'>{productName}</div>

        <div className='flex justify-between items-center mt-6'>
            <div className='text-[11px] text-stone-500 font-semibold'>{size}</div>
            
        </div>

        <div className='absolute right-4 bottom-4 text-3xl font-bold'>{price}$</div>
        <div className='absolute left-4 bottom-4 text-md font-bold'>{love_count} ❤️</div>
      </div>
    </div>
  )
}
